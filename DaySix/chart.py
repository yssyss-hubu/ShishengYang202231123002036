import matplotlib.pyplot as plt
import networkx as nx

layers = {
    0: "Input\n[3x640x640]",
    1: "Conv\n[3→16, k3, s2]",
    2: "Conv\n[16→32, k3, s2]",
    3: "C2f\n[32→32, d=1]",
    4: "Conv\n[32→64, k3, s2]",
    5: "C2f\n[64→64, d=2]",
    6: "Conv\n[64→128, k3, s2]",
    7: "C2f\n[128→128, d=2]",
    8: "Conv\n[128→256, k3, s2]",
    9: "C2f\n[256→256, d=1]",
    10: "SPPF\n[256→256]",
    11: "Upsample x2",
    12: "Concat\n(+128)",
    13: "C2f\n[384→128]",
    14: "Upsample x2",
    15: "Concat\n(+64)",
    16: "C2f\n[192→64]",
    17: "Conv\n[64→64, s2]",
    18: "Concat\n(+128)",
    19: "C2f\n[192→128]",
    20: "Conv\n[128→128, s2]",
    21: "Concat\n(+256)",
    22: "C2f\n[384→256]",
    23: "Detect\n[83 cls]"
}

# 初始化图
G = nx.DiGraph()
for idx, label in layers.items():
    G.add_node(idx, label=label)

# 添加边（模块之间的连接）
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (6, 12), (12, 13),
    (13, 14), (14, 15), (4, 15), (15, 16),
    (16, 17), (13, 18), (17, 18), (18, 19),
    (19, 20), (10, 21), (20, 21), (21, 22),
    (16, 23), (19, 23), (22, 23)
]
G.add_edges_from(edges)

# 布局并绘制
pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(20, 14))
nx.draw(G, pos, with_labels=False, node_size=2500, node_color='lightgreen', edge_color='gray', arrows=True)
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=9)
plt.title("YOLOv8n 网络结构流程图", fontsize=16)
plt.axis('off')
plt.show()
#