#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, b) for(int i = a; i < b; ++i)
#define tr(a, x) for(auto& a : x)
#define all(x) x.begin(), x.end()
#define sz(x) (int)(x).size()
#define w(a) while(a--)
#define cint(n) int n; cin >> n
#define endl '\n'
#define fastio ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
typedef long long ll;
typedef pair<int, int> pi;
typedef vector<int> vi;

int main() {
    fastio;

    ifstream fin("graph.txt");
    vector<pi> edges;
    int a, b;
    int max_node = 0;
    while (fin >> a) {
        if (a == -1) break;
        fin >> b;
        edges.emplace_back(a, b);
        max_node = max({max_node, a, b});
    }
    int n = max_node + 1;

    vector<vi> adj[2];
    adj[0].resize(n);
    adj[1].resize(n);
    for (auto [u, v] : edges) {
        adj[0][u].push_back(v);
        adj[1][v].push_back(u);
    }

    using State = tuple<ll, int, int>;
    priority_queue<State, vector<State>, greater<State>> pq;
    vector<vector<ll>> dp(n, vector<ll>(2, LLONG_MAX));
    dp[0][0] = 0;
    pq.emplace(0, 0, 0);

    while (!pq.empty()) {
        auto [cost, u, s] = pq.top();
        pq.pop();

        if (u == n - 1) {
            cout << cost << endl;
            return 0;
        }

        if (cost > dp[u][s]) continue;

        for (int v : adj[s][u]) {
            ll new_cost = cost + 1;
            if (new_cost < dp[v][s]) {
                dp[v][s] = new_cost;
                pq.emplace(new_cost, v, s);
            }
        }

        int new_s = 1 - s;
        ll new_cost = cost + n;
        if (new_cost < dp[u][new_s]) {
            dp[u][new_s] = new_cost;
            pq.emplace(new_cost, u, new_s);
        }
    }

    ll ans = min(dp[n-1][0], dp[n-1][1]);
    if (ans == LLONG_MAX) {
        cout << -1 << endl;
    } else {
        cout << ans << endl;
    }

    return 0;
}
