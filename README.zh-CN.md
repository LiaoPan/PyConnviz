# PyConnviz

[English](README.md) | 简体中文

PyConnviz 0.1.0 是一个以 Nilearn 为优先、面向出版级绘图的 Python 库，用于
可视化已经计算完成的 MEG/EEG 源空间 ROI 功能连接。它可以生成静态及交互式
皮层表面网络、Nilearn 玻璃脑与 HTML 连接组，以及 MNE-Connectivity 环形图。

PyConnviz 不计算功能连接、不进行源定位或统计检验、不替用户选择显著连接，
也不会下载模板。它不包含 GUI 或 CLI。

## 效果展示

以下图片由 PyConnviz 根据仓库中已经过验证的示例生成。推荐的 Plotly 和
玻璃脑预览使用确定性的模拟连接数据；固定视角的 Matplotlib 和原生 Nilearn
图片使用投影到 fsaverage 的真实 MSDL 连接数据。

<table>
  <thead>
    <tr>
      <th>推荐：Plotly surface 交互式预览</th>
      <th>推荐：Nilearn 玻璃脑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><img src="docs/images/surface-plotly.png" width="100%" alt="PyConnviz Plotly surface 连接组四视图"></td>
      <td width="50%"><img src="docs/images/glass-brain.png" width="100%" alt="PyConnviz Nilearn 玻璃脑连接组"></td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>补充：Matplotlib surface 三视图</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="docs/images/surface-matplotlib.png" width="100%" alt="包含左右外侧与全脑背侧视图的 PyConnviz 深度感知 Matplotlib surface 连接组"></td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>补充：原生 Nilearn surface 六视图</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="docs/images/surface-nilearn.png" width="100%" alt="包含外侧、内侧和背侧视图的 PyConnviz 半透明原生 Nilearn surface 连接组"></td>
    </tr>
  </tbody>
</table>

<p align="center">
  <img src="docs/images/circle.png" width="60%" alt="PyConnviz MNE-Connectivity 环形图">
</p>

## 安装

PyConnviz 需要 Python 3.10 或更高版本。

### 从 PyPI 安装

从 [PyPI](https://pypi.org/project/pyconnviz/) 安装最新正式版本：

```bash
python -m pip install --upgrade pyconnviz
python -m pip install --upgrade "pyconnviz[interactive]"
python -m pip install --upgrade "pyconnviz[interactive,export]"
```

第一条命令安装核心静态渲染器。交互式 Plotly HTML surface 使用
`interactive`；如还需通过 Kaleido 导出 Plotly 静态 PNG/SVG/PDF，请同时
安装 `interactive,export`。

从源码检出安装或配置发布开发环境时：

```bash
python -m pip install -e .
python -m pip install -e ".[dev,interactive,export]"
```

若要获得完全可复现的验收环境，请使用 `-c constraints-dev.txt` 安装。导入
`pyconnviz` 不会访问网络、下载数据、导入 Plotly 或打开绘图窗口。

## 推荐视图

请根据希望回答的问题选择视图：

1. Plotly surface 是**首选的交互式解剖视图**。它保留 surface-RAS 几何，
   可以自由旋转，并可用于判断连接位于皮层前方还是后方。
2. 在具有有效 MNI 坐标时，Nilearn 玻璃脑是**首选的静态连接概览**。它不受
   皮层遮挡影响，最便于比较完整的全脑连接图。
3. Matplotlib 和原生 Nilearn surface 是**补充性的固定视角解剖背景**。
   半透明皮层与相机深度 alpha 提示能够改善空间判读，但固定的二维投影并非
   真实的三维管状渲染，也不能替代可旋转的 Plotly 视图。

对应调用按推荐顺序如下：

```python
plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="plotly",
    output="surface_interactive.html",
)
plot_connectome(
    prepared,
    geometry,
    backend="glass",
    output="glass_brain.svg",
)
plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="matplotlib",
    depth_cue=True,
    output="surface_context.png",
)
```

为保持向后兼容，分发器仍默认使用依赖较少的 Matplotlib 渲染器。这里的
“推荐”只是展示方式的选择，不会改变连接筛选规则或公共默认行为。

## 快速开始

下面这个可运行示例使用一个很小的模拟双侧网格，因此无需下载数据。真实皮层
表面通常来自 FreeSurfer。几何对象负责节点顺序与坐标空间，连接矩阵必须使用
相同的节点顺序。

```python
import numpy as np
from pyconnviz import (
    HemisphereMesh,
    geometry_from_arrays,
    plot_connectome,
    prepare_connectome,
)

faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
left_xyz = np.array([[-4, -1, -1], [-2, -1, 1], [-4, 1, 1], [-2, 1, -1]])
right_xyz = left_xyz + [6, 0, 0]
geometry = geometry_from_arrays(
    ("L-a", "L-b", "R-a", "R-b"),
    np.array([left_xyz[0], left_xyz[3], right_xyz[0], right_xyz[3]]),
    ("left", "left", "right", "right"),
    {
        "left": HemisphereMesh(left_xyz, faces),
        "right": HemisphereMesh(right_xyz, faces),
    },
    mni_coords=np.array([
        [-30, -10, 20], [-20, 10, 20], [20, -10, 20], [30, 10, 20]
    ]),
    node_vertices=np.array([0, 3, 0, 3]),
)
matrix = np.array([
    [0.0, 0.4, -0.8, 0.0],
    [0.4, 0.0, 0.6, 0.0],
    [-0.8, 0.6, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
])
prepared = prepare_connectome(
    matrix,
    geometry=geometry,
    max_edges=4,
)

# Recommended rotatable anatomical view (install pyconnviz[interactive]).
interactive = plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="plotly",
    output="quickstart_surface.html",
)

# Recommended static overview when geometry.mni_coords is valid MNI space.
glass = plot_connectome(
    prepared,
    geometry,
    backend="glass",
    output="quickstart_glass.svg",
)

# Base-install fallback and supplementary fixed anatomical context.
surface = plot_connectome(
    prepared,
    geometry,
    engine="matplotlib",
    depth_cue=True,
    output="quickstart_surface.png",
)
print(interactive.output_files, glass.output_files, surface.output_files)
print(prepared.edges)  # exact same immutable edges in every figure
```

`examples/01_numpy_surface.py` 是基础安装环境中的静态版本；
`examples/04_all_backends.py` 会从同一个已准备网络导出所有渲染器的结果。

## 所有绘图模式

每个渲染器都消费同一个不可变的 `PreparedConnectome`；后端不会再次对矩阵进行
阈值筛选或排序。

| 后端 | 引擎 | 所需坐标 | 输出 | 额外安装项 |
|---|---|---|---|---|
| `surface` | `plotly` | surface-RAS + surface 网格 | 离线 HTML；通过 Kaleido 输出 PNG、SVG、PDF | `interactive`；静态文件需要 `export` |
| `glass` | Nilearn | MNI 毫米坐标 | PNG、SVG、PDF | 无 |
| `html` | Nilearn `view_connectome` | MNI 毫米坐标 | 离线 HTML | 无 |
| `surface` | `matplotlib` | surface-RAS + surface 网格 | PNG、SVG、PDF | 无 |
| `surface` | `nilearn`（`plot_img_on_surf`） | 已配准的 pial/inflated 网格 | PNG、SVG、PDF | 无 |
| `circle` | MNE-Connectivity | 仅需节点顺序 | PNG、SVG、PDF | 无 |

只需准备一次，即可选择任意输出，且不会改变科学连接集合：

```python
from pyconnviz import plot_connectome, prepare_connectome

prepared = prepare_connectome(
    matrix,
    geometry=geometry,
    edge_threshold="75%",
    max_edges=120,
)

plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="plotly",
    output="surface_interactive.html",
)
plot_connectome(prepared, geometry, backend="glass", output="glass_brain.svg")
plot_connectome(prepared, geometry, backend="html", output="connectome.html")
plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="matplotlib",
    depth_cue=True,
    output="surface_context.png",
)
plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="nilearn",
    views=["lateral", "medial", "dorsal"],
    hemispheres=["left", "right"],
    depth_cue=True,
    output="surface_nilearn_context.png",
)
plot_connectome(prepared, geometry, backend="circle", output="circle.png")
```

在相同调用中指定 `style="soft"` 或 `style="dark"`，即可选择其他内置视觉
样式。`examples/04_all_backends.py` 是完整的离线导出示例。

## NumPy 连接数据与显式几何

二维 NumPy 输入必须是节点数 × 节点数的方阵。对称矩阵表示无向连接；当
`directed="auto"` 时，非对称矩阵表示有向连接。PyConnviz 不会擅自猜测三角
NumPy 矩阵的含义；只有当这符合所存数据的科学含义时，才应设置
`symmetrize="lower"` 或 `symmetrize="upper"`。

对于更高维数组，需要同时指定两个节点轴以及每个待归约轴：

```python
prepared = prepare_connectome(
    connectivity_by_epoch_frequency_node_node,
    geometry=geometry,
    node_axes=(-2, -1),
    reduce_axes=(0, 1),
    reduction="mean",
    edge_mask=corrected_p < 0.05,
    edge_threshold=0.20,
    max_edges=120,
)
```

数组轴绝不会被静默展平。只有当复数矩阵具有明确的科学解释时，才应显式选择
`complex_mode="magnitude"`、`"real"` 或 `"imag"`。

## MNE 标签与 FreeSurfer 几何

PyConnviz 会严格保留标签顺序。请在构建几何对象之前筛选或重新排列标签，确保
其顺序与连接矩阵的节点顺序完全一致。

```python
import mne
from pyconnviz import geometry_from_mne_labels

labels = mne.read_labels_from_annot(
    subject="fsaverage", parc="aparc", subjects_dir=subjects_dir
)
labels = [
    label for label in labels
    if "unknown" not in label.name.lower()
    and "corpuscallosum" not in label.name.lower()
]
geometry = geometry_from_mne_labels(
    labels,
    subject="fsaverage",
    subjects_dir=subjects_dir,
    src=src,
    surface="inflated",
    sphere_surface="sphere",
)
```

`examples/02_mne_labels_surface.py` 使用用户提供的本地 FreeSurfer 被试数据。
PyConnviz 不会下载 `fsaverage` 或任何其他被试数据。

## MNE-Connectivity 频率、时间与 epoch 选择

常见的双变量 MNE-Connectivity 对象通过其公共 API
`get_data(output="dense")`、`dims`、`coords`、`names` 和 `n_nodes` 进行适配。

```python
result = plot_connectome(
    spectral_connectivity,
    geometry,
    freq=(8.0, 13.0),
    reduction="mean",
    edge_mask=corrected_p < 0.05,
    max_edges=120,
    output="alpha_surface.svg",
)
```

标量频率或时间会选择最近的公共坐标。包含两个值的范围会按闭区间选择所有坐标，
然后按 `mean` 或 `median` 归约。非单元素的频率、时间或 epoch 维度必须提供显式
选择器。多变量组件维度绝不会被静默展平。

MNE-Connectivity 0.9.0 将经过审计的全对全双变量谱连接结果存储在其公共稠密
表示的下三角中。在默认 `symmetrize="auto"` 路径下，只有当 `indices is None`
并且公共 `method` 属于白名单中的对称标量指标（`coh`、`plv`、`ciplv`、
`ppc`、`pli`、`pli2_unbiased`、`wpli` 或 `wpli2_debiased`）时，PyConnviz 才会
重建完整无向矩阵。`dpli` 等有向方法、显式索引对、Granger 指标和复数相干性
不会被静默镜像。推断依据会保存在结果元数据中。

## 已准备数据与渲染器不变量

科学连接的选择只发生在 `prepare_connectome` 中。渲染器接收同一个不可变的
`PreparedConnectome`，不能再次执行阈值筛选或排序。

```python
from pyconnviz import plot_connectome, prepare_connectome

prepared = prepare_connectome(
    connectivity,
    geometry=geometry,
    freq=(8.0, 13.0),
    edge_mask=corrected_p < 0.05,
    max_edges=120,
)
interactive = plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="plotly",
    static_views=("left", "right", "dorsal", "ventral"),
    image_width=1400,
    image_height=1000,
    output=("alpha_surface.html", "alpha_surface.png"),
)
glass = plot_connectome(prepared, geometry, backend="glass", output="alpha_glass.svg")
surface = plot_connectome(
    prepared,
    geometry,
    engine="matplotlib",
    depth_cue=True,
    output="alpha_surface_context.png",
)
circle = plot_connectome(prepared, geometry, backend="circle", output="alpha_circle.png")
native_html = plot_connectome(
    prepared,
    geometry,
    backend="html",  # direct nilearn.plotting.view_connectome wrapper
    output="alpha_connectome.html",
)
```

## 原生 Nilearn pial 拼图

使用 `engine="nilearn"` 可生成原生 `plot_img_on_surf` pial 拼图。纯连接绘图应
省略 `stat_map`；只有当统计体数据确实属于当前分析时才应传入。可以直接传入
FreeSurfer 被试目录；PyConnviz 会验证其中的 pial、inflated、sulcal 和 curvature
文件，并且绝不会下载替代数据。

```python
native_surface = plot_connectome(
    prepared,
    pial_geometry,  # node_vertices index the same full fsaverage topology
    backend="surface",
    engine="nilearn",
    surf_mesh="data/fsaverage",
    views=["lateral", "medial", "dorsal"],
    hemispheres=["left", "right"],
    bg_on_data=True,
    symmetric_cmap=None,
    symmetric_cbar="auto",
    inflate=False,
    depth_cue=True,
    output="native_surface_three_views.png",
)
```

Nilearn 会对视角模式和半球做笛卡尔积，因此这个示例生成三行视角、两列半球，
共六个皮层面板。单半球 lateral 面板会显示其完整的半球内已准备连接子集；它们
不会、也不应显示跨半球连接。当跨半球连接必须出现在同一视图中时，请使用
Matplotlib 的自定义 `both` 面板、可旋转 Plotly surface、玻璃脑或原生
`view_connectome` HTML 后端。

在这个纯连接调用中，原生引擎会提供一个内部透明的全零 Niimg，仅仅因为
`plot_img_on_surf` 要求输入一个体数据。可见皮层保持中性，不携带任何推断出的
功能数值。逐顶点 `surface_values` 和功能性 `node_overlay` 模式仍由 Matplotlib
或 Plotly 引擎处理；PyConnviz 绝不会从顶点值重建体数据。

对于单独且明确标注的体到表面分析，仍可显式传入 `stat_map=stat_img`。此时颜色
和统计色条表示所提供的图像，而不是连接矩阵。

如果输入已经准备好的对象，再传入 `edge_threshold`、`edge_mask`、`max_edges`、
`freq`、`time` 或其他科学参数会报错。

## 节点与连接的视觉编码

节点颜色可以通过 `node_color_by="strength"`、`"hemisphere"`、`"group"` 或
`"custom"` 选择；按分组着色需要 `geometry.groups`。节点大小接受
`node_size_by="strength"` 或 `"custom"`。可以通过 `node_values` 提供共用的
自定义向量；若颜色和大小必须使用不同的有限数值向量，则分别使用
`node_color_values` 和 `node_size_values`。这些选项只控制显示映射，绝不会改变
`PreparedConnectome.edges`。

`edge_cmap="auto"` 会保留有符号数据与单侧数据的颜色语义。显式指定 Matplotlib
色图名称只会覆盖显示调色板，并会一致地转发到 surface、glass、HTML 和 circle
后端。

## 样式与自定义视图

`style="paper"`、`style="soft"` 和 `style="dark"` 是完整的视觉预设。可以查看
解析后参数的副本，而不会改变全局状态：

```python
from pyconnviz import ViewSpec, get_style, plot_connectome

print(get_style("paper"))
custom_views = (
    ViewSpec("left", "lateral", "Left lateral"),
    ViewSpec("right", "lateral", "Right lateral"),
    ViewSpec("both", (80.0, -90.0), "Whole-brain dorsal"),
)
plot_connectome(
    prepared,
    geometry,
    backend="surface",
    engine="matplotlib",
    views=custom_views,
    style="paper",
    output="custom_views.svg",
)
```

Matplotlib surface 引擎支持 `views="paper"`、`views="four"` 和 `views="whole"`
等字符串预设。原生 Nilearn 引擎接受 Nilearn 视角字符串；Plotly 使用 `view`
设置初始相机，并用 `static_views` 配置多视角静态导出。

表面叠加（surface overlay）具有独立的 `surface_cmap`（默认 `"RdBu_r"`），而 `node_cmap`
控制节点标记。改变任一调色板都只影响显示，不能改变已准备连接集合。每个内置
样式都默认使用中性灰度皮层和 `node_overlay="none"`；只有提供
`surface_values` 或显式的非 `"none"` overlay 时，才会创建功能性皮层色图。
没有 overlay 时，PyConnviz 仍会通过 Nilearn 发送一个透明 surface map，使配置
的皮层亮度和脑沟背景保持可见，而不是退化成白色网格。

显式的 `cortex_alpha` 控制三个 surface 引擎中的皮层不透明度，其值必须是
`0` 到 `1` 之间的有限数。未显式指定时，Plotly 保留交互式样式默认值：
`paper / soft / dark` 分别为 `0.24 / 0.20 / 0.32`。Matplotlib 和原生 Nilearn
使用更加透明的固定视角静态默认值 `0.08 / 0.08 / 0.18`，使内部连接能在固定
投影中保持可见。它们也使用渲染器专属的默认连接可见性预算：paper 的 alpha
为 `0.95`、线宽范围为 `1.4–5.4 pt`、投影深度因子下限为 `0.70`。显式传入
`edge_alpha`、`edge_width_range` 或 `cortex_alpha` 时始终优先于这些视觉默认值。

两个固定视角引擎都默认启用 `depth_cue=True`：每条已准备曲线会被分割为多个
显示线段，只有 alpha 会随投影后的相机深度变化。较近线段更强，较远线段更淡，
但提高后的下限可以避免有效连接在 README 缩略图尺度下消失。连接颜色仍编码
符号或幅值，连接宽度仍编码绝对权重，并且每个面板都保留其完整的半球范围连接
集合。设置 `depth_cue=False` 可使用统一线条透明度。

## 坐标约定：surface-RAS 与 MNI

`ConnectomeGeometry.surface_coords` 属于当前显示的 surface 网格，通常位于
FreeSurfer surface-RAS 空间，例如个体被试的 inflated surface。只有 Matplotlib
和 Plotly surface 后端使用该字段。

`ConnectomeGeometry.mni_coords` 包含以毫米为单位的 MNI 坐标。只有 Nilearn
glass 和 HTML 后端使用该字段。inflated surface 坐标绝不会替代 MNI 坐标；缺少
所需坐标字段时会立即报错。

`min_distance_mm` 当前使用 `surface_coords` 之间的欧氏距离。它是显示筛选条件，
不表示测地距离或解剖距离。

## 掩码、阈值、百分位数与 Top-K 语义

精确的筛选顺序如下：

1. 丢弃对角线和非有限候选值；
2. 应用 `edge_mask`；
3. 应用 `keep_sign`；
4. 应用 `min_distance_mm`；
5. 应用 `edge_threshold`；
6. 按绝对权重、源节点、目标节点进行稳定排序；
7. 应用 `max_edges`。

`edge_mask` 表示在外部计算出的合格性或统计显著性。数值型 `edge_threshold`
保留 `abs(weight) >= threshold` 的连接；`"95%"` 等字符串只从通过前面筛选的
候选项计算百分位数。`max_edges` 是确定性的可视化数量上限。这些概念被有意
分开处理。

## 复数、有符号、非负、三角与有向数据

默认情况下，复数输入会报错。必须显式选择 `complex_mode="magnitude"`、
`"real"` 或 `"imag"`。PyConnviz 绝不会根据指标名称自行推断转换方式。

当可见连接同时包含正值和负值时，渲染器使用以零为中心的发散色阶。非负的
coherence/wPLI 类数据使用顺序色阶，而不是容易造成误导的红蓝有符号色阶。

PyConnviz 不会猜测三角 NumPy 矩阵是否为无向矩阵。请显式使用
`symmetrize="lower"` 或 `symmetrize="upper"`。在 `directed="auto"` 模式下，
非对称矩阵被视为有向矩阵，并保留源节点到目标节点的顺序：

```python
directed = np.array([[0.0, 0.8], [-0.2, 0.0]])
result = plot_connectome(
    directed,
    geometry,
    directed=True,
    max_edges=2,
    output="directed_surface.png",
    show_arrows=True,
)
```

## 表面叠加（Surface overlay）

`node_overlay` 可以是 `"none"`、`"roi"`、`"gaussian"` 或 `"vertex"`。默认值
始终是 `"none"`，包括 `style="soft"`。

- ROI 模式将节点值赋给显式的 `roi_vertices`。
- Gaussian 模式通过半径截断的稀疏网格图扩散数值，绝不跨越半球。若要使用
  欧氏距离，必须显式指定。
- Vertex 模式显示用户提供的真实逐顶点数值，并具有最高优先级。

Gaussian overlay 只是视觉插值，不代表源重建、皮层传播或统计显著性。

## 交互式与静态导出细节

`show=False` 是默认值，并支持在无显示环境中使用 `MPLBACKEND=Agg`。Plotly
HTML 默认嵌入 JavaScript，因此可以离线打开。`image_width`、`image_height` 和
`image_scale` 控制 Plotly 静态导出，但不会改变交互数据。Plotly 静态输出默认
为一个双列拼图，其中包含 `left`、`right`、`dorsal` 和 `ventral` 场景；可以用
`static_views` 自定义顺序，或传入 `static_views=None`，只导出单个交互式 `view`
相机。

两个交互式输出回答不同的问题。Plotly `surface` 后端保留皮层解剖结构并支持
显式逐顶点 overlay；原生 Nilearn `html` 后端调用 `view_connectome`，由于没有
遮挡性的皮层网格，通常更适合旋转和检查完整连接图。二者使用相同的已准备连接
矩阵，但 surface 渲染使用 surface-RAS 坐标，而 `view_connectome` 需要 MNI
毫米坐标。

验收生成器同时使用 Nilearn 内置的 `fsaverage5` 皮层，以及当前工作区提供的
全分辨率 `data/fsaverage` 被试数据。原生验收拼图使用具有 163,842 个顶点的 pial
网格和中性的无 `stat_map` 路径，因此其皮层没有无关的统计投影。生成器还会
同时输出 `surface_interactive.html` 与 1400×1000 的四视图
`surface_interactive.png`，并通过与 `surface_paper.png` 相同的半透明、深度感知
pial 渲染器刷新 `surface_fsaverage_full.png`。库的绘图调用仍要求显式提供几何
数据，并且绝不会静默替换或下载模板。

## 构建与发布

`pyproject.toml` 是包元数据和构建配置的唯一来源。PyConnviz 使用 PEP 517
`setuptools.build_meta` 后端，并且特意不提供 `setup.py` 入口。

只需安装一次发布工具：

```bash
python -m pip install -e ".[dev]"
```

在仓库根目录使用统一发布脚本。每个操作都会在全新的临时目录中构建，验证刚刚
生成的 wheel 和 sdist，然后将其复制到 `dist/`，再停止或上传：

```bash
./scripts/release.sh build       # build + twine check; never uploads
./scripts/release.sh testpypi    # build + check + TestPyPI upload
./scripts/release.sh pypi        # build + check + confirmed production upload
```

不带操作参数运行 `./scripts/release.sh` 等价于 `build`。正式发布操作会要求输入
`release pyconnviz`；非交互式发布自动化必须通过
`./scripts/release.sh pypi --yes` 显式确认。Twine 从标准环境变量、keyring 或
配置文件中获取凭据，脚本绝不会保存凭据。设置 `PYCONNVIZ_PYTHON` 可以覆盖
默认的 `.venv/bin/python` 发现逻辑。

排查问题时，也可以手动运行构建操作所执行的现代 PEP 517 步骤：

```bash
python -m build
python -m twine check dist/*
```

生成的文件为 `dist/pyconnviz-0.1.0.tar.gz` 和
`dist/pyconnviz-0.1.0-py3-none-any.whl`。执行 `testpypi` 操作后，请先安装并检查
该版本，再发布到正式索引：

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  pyconnviz
```

两个上传操作都会改变外部状态，并需要维护者凭据。脚本只上传当前临时暂存目录中
刚刚创建的产物，绝不会使用可能包含旧版本的宽泛 `dist/*` glob。

效果展示使用 `docs/images` 中的仓库相对路径。在发布 PyPI 之前，请配置公开的
项目/仓库 URL；如果目标 PyPI 渲染器不能显示相对图片，请把这五处图片引用改为
由公开仓库托管的绝对 URL。

## 真实上游准确性审计

验证脚本会运行两个真实数据 Nilearn 用例（MSDL atlas correlation，以及通过
`GraphicalLassoCV` 提取的 Power-264 sphere）、两个独立计算的 MNE-Connectivity
用例（`coh` 和 `plv`），以及一个 MSDL 到 fsaverage 的 surface 用例。它会保留
五个官方 Python 示例及其 URL 和 SHA-256 摘要，计算独立的连接/强度判定基准，
并在 `artifacts/upstream_validation` 下写入 JSON、Markdown、PNG、HTML、NumPy
数组、逐用例 manifest 和带标签的对比图。

对于每个匹配渲染器的比较，上游绘图函数与 PyConnviz 会接收相同的已准备矩阵、
坐标、节点样式、连接样式，以及被禁用的渲染器侧阈值。这样可以将科学数据一致性
与上游默认阈值差异分开。原生默认输出仍作为诊断证据保留，而不会被当作完全匹配
的渲染结果。

第一次运行需要联网获取官方源码、MSDL atlas 和 development-fMRI 输入：

```bash
MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/python scripts/validate_upstream_examples.py \
  --data-dir data/upstream_validation \
  --outdir artifacts/upstream_validation \
  --fsaverage-dir data/fsaverage
```

缓存完成后，可以使用 `--skip-download` 在不联网的情况下重复审计。当自动化必须
在任一必需差异出现时返回非零状态，请添加 `--strict`。严格模式的非零状态表示
审计发现了科学不一致；应查看 `report.md`，而不是将其视为验证进程崩溃。surface
用例会把每个 MSDL MNI 中心映射到同半球最近的全分辨率 fsaverage pial 顶点，
记录每个距离，排除超过 10 mm 的中心，并把同一组 top-40 连接传给 Matplotlib、
原生 Nilearn 和 Plotly surface 引擎。这是透明的可视化坐标审计，并不声称具有
个体特异的解剖配准准确性。

## 常见错误

- “requires geometry.mni_coords”：为 glass/HTML 提供真实的 MNI 毫米坐标；不要
  复制 surface-RAS 数值。
- “requires node_axes”：高维 NumPy 输入必须显式指定两个节点轴和所有
  `reduce_axes`。
- “complex connectivity”：显式选择 `complex_mode`。
- “matrix must be symmetric”：选择 `directed=True` 或显式的 `symmetrize` 规则。
- 缺少 Plotly/Kaleido：分别安装 `pyconnviz[interactive]` 或
  `pyconnviz[export]`。
- 缺少 FreeSurfer 文件：在 `subjects_dir` 下准备被试数据；核心 API 绝不会发起
  下载。

## v0.1 已知限制

Matplotlib 和原生 Nilearn 的连接是三维线条，而不是真实管道。它们随视角变化的
alpha 只是光学深度提示，并不会执行真实的网格遮挡，也不得被解释为额外的数据
变量。单半球面板会省略跨半球连接，全脑面板则包含这些连接。Plotly 每条连接
使用一个 trace，因此大型连接集合会生成较大的 HTML 文件。首个版本专注于皮层
ROI，尚不提供完整的混合源或皮层下 surface 渲染。
