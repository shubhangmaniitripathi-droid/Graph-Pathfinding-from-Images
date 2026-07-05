import cv2
import numpy as np
import pytesseract
import networkx as nx
from ultralytics import YOLO

image_path = r"C:\Users\ADITYA\Downloads\graph_images\graph_images\graph_391.png"
model_path = r"C:\Users\ADITYA\runs\detect\Strat 2\weights\best.pt"
output_image_path = "final_graph_result.png"
output_graph_txt = "graph.txt"

DISTANCE_THRESHOLD = 35
ARROW_MATCH_THRESHOLD = 25
MERGE_IOU_THRESHOLD = 0.2  

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
    param1=50, param2=30, minRadius=10, maxRadius=40
)

node_ids = {}
if circles is not None:
    circles = circles[0].astype("int")
    for (x, y, r) in circles:
        margin = int(r * 0.8)
        x1, y1 = max(x - margin, 0), max(y - margin, 0)
        x2, y2 = min(x + margin, image.shape[1]), min(y + margin, image.shape[0])
        roi = gray[y1:y2, x1:x2]
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        text = pytesseract.image_to_string(roi, config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()
        if text.isdigit():
            node_ids[(x, y)] = int(text)

id_to_coords = {v: k for k, v in node_ids.items()}

model = YOLO(model_path)
results = model(image_path, imgsz=960)[0]

label_map = {0: "arrow", 1: "edge"}
edges = []
arrows = []

edge_counter = 1
arrow_counter = 1

for box in results.boxes.data.tolist():
    x1, y1, x2, y2, conf, cls = box[:6]
    cls = int(cls)
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    if cls == 0:
        arrows.append((arrow_counter, [x1, y1, x2, y2]))
        arrow_counter += 1
    elif cls == 1:
        edges.append((edge_counter, [x1, y1, x2, y2]))
        edge_counter += 1

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def merge_boxes(boxes):
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return [x1, y1, x2, y2]

merged_edges = []
used = [False] * len(edges)
for i in range(len(edges)):
    if used[i]:
        continue
    group = [edges[i][1]]
    used[i] = True
    for j in range(i+1, len(edges)):
        if used[j]:
            continue
        if compute_iou(edges[i][1], edges[j][1]) > MERGE_IOU_THRESHOLD:
            group.append(edges[j][1])
            used[j] = True
    merged_bbox = merge_boxes(group)
    merged_edges.append((len(merged_edges) + 1, merged_bbox))

def node_within_range(point, node_coords, max_dist):
    candidates = []
    for nid, coords in node_coords.items():
        dist = np.linalg.norm(np.array(coords) - np.array(point))
        if dist <= max_dist:
            candidates.append((dist, nid))
    if candidates:
        return min(candidates)[1]
    return None

def process_edges(edges_list, id_to_coords, arrows, img):
    existing_edges = set()
    G_local = nx.DiGraph()
    G_local.add_nodes_from(id_to_coords.keys())

    for edge_idx, (x1, y1, x2, y2) in edges_list:
        diag1 = [(x1, y1), (x2, y2)]
        diag2 = [(x1, y2), (x2, y1)]

        n1_diag1 = node_within_range(diag1[0], id_to_coords, DISTANCE_THRESHOLD)
        n2_diag1 = node_within_range(diag1[1], id_to_coords, DISTANCE_THRESHOLD)
        n1_diag2 = node_within_range(diag2[0], id_to_coords, DISTANCE_THRESHOLD)
        n2_diag2 = node_within_range(diag2[1], id_to_coords, DISTANCE_THRESHOLD)

        valid_diag1 = n1_diag1 is not None and n2_diag1 is not None
        valid_diag2 = n1_diag2 is not None and n2_diag2 is not None

        if not valid_diag1 and not valid_diag2:
            continue

        diag1_dist = float('inf')
        diag2_dist = float('inf')

        if valid_diag1:
            diag1_dist_1 = np.linalg.norm(np.array(id_to_coords[n1_diag1]) - np.array(diag1[0]))
            diag1_dist_2 = np.linalg.norm(np.array(id_to_coords[n2_diag1]) - np.array(diag1[1]))
            diag1_dist = diag1_dist_1 + diag1_dist_2
        if valid_diag2:
            diag2_dist_1 = np.linalg.norm(np.array(id_to_coords[n1_diag2]) - np.array(diag2[0]))
            diag2_dist_2 = np.linalg.norm(np.array(id_to_coords[n2_diag2]) - np.array(diag2[1]))
            diag2_dist = diag2_dist_1 + diag2_dist_2

        if diag1_dist <= diag2_dist:
            node_a, node_b = n1_diag1, n2_diag1
            selected_diag = "positive slope (\\)"
            dist1, dist2 = diag1_dist_1, diag1_dist_2
        else:
            node_a, node_b = n1_diag2, n2_diag2
            selected_diag = "negative slope (/)"
            dist1, dist2 = diag2_dist_1, diag2_dist_2

        if (node_a, node_b) in existing_edges or (node_b, node_a) in existing_edges:
            continue

        arrows_near_a = []
        arrows_near_b = []

        for arrow_idx, (ax1, ay1, ax2, ay2) in arrows:
            arrow_cx = (ax1 + ax2) // 2
            arrow_cy = (ay1 + ay2) // 2

            if min(x1, x2) <= arrow_cx <= max(x1, x2) and min(y1, y2) <= arrow_cy <= max(y1, y2):
                corner_a = (x1, y1) if selected_diag == "positive slope (\)" else (x1, y2)
                corner_b = (x2, y2) if selected_diag == "positive slope (\)" else (x2, y1)

                dist_to_a = np.linalg.norm(np.array([arrow_cx, arrow_cy]) - np.array(corner_a))
                dist_to_b = np.linalg.norm(np.array([arrow_cx, arrow_cy]) - np.array(corner_b))

                if dist_to_a <= ARROW_MATCH_THRESHOLD:
                    arrows_near_a.append(arrow_idx)
                if dist_to_b <= ARROW_MATCH_THRESHOLD:
                    arrows_near_b.append(arrow_idx)

        if arrows_near_a and not arrows_near_b:
            G_local.add_edge(node_b, node_a)
            existing_edges.add((node_b, node_a))
        elif arrows_near_b and not arrows_near_a:
            G_local.add_edge(node_a, node_b)
            existing_edges.add((node_a, node_b))
        elif arrows_near_a and arrows_near_b:
            G_local.add_edge(node_a, node_b)
            G_local.add_edge(node_b, node_a)
            existing_edges.add((node_a, node_b))
            existing_edges.add((node_b, node_a))
        else:
            continue

        cv2.line(img, id_to_coords[node_a], id_to_coords[node_b], (0, 255, 0), 2)

    return G_local

G_before = process_edges(edges, id_to_coords, arrows, image)
G_after = process_edges(merged_edges, id_to_coords, arrows, image)

G_final = nx.DiGraph()
G_final.add_nodes_from(id_to_coords.keys())
G_final.add_edges_from(G_before.edges)
G_final.add_edges_from(G_after.edges)

for nid, (cx, cy) in id_to_coords.items():
    cv2.circle(image, (cx, cy), 20, (0, 0, 255), 2)
    cv2.putText(image, str(nid), (cx - 15, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.imwrite(output_image_path, image)

with open(output_graph_txt, "w") as f:
    for u, v in G_final.edges():
        f.write(f"{u} {v}\n")
    f.write("-1")
