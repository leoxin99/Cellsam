"""
SarcGraph驱动的提示框生成器
利用Z线检测 + DBSCAN聚类生成高质量的SAM提示框

使用方法:
    from prompt_generator import SarcGraphPromptGenerator
    
    generator = SarcGraphPromptGenerator(pixel_size_um=0.5)
    boxes = generator.generate_prompts(actinin_image)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, HDBSCAN
from skimage.feature import blob_log, blob_dog
from scipy.ndimage import gaussian_filter
import warnings


@dataclass
class BoundingBox:
    """边界框数据类"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 1.0
    num_zlines: int = 0
    cluster_id: int = -1
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """返回 (x1, y1, x2, y2) 格式"""
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """返回 (x, y, width, height) 格式"""
        return (self.x_min, self.y_min, 
                self.x_max - self.x_min, self.y_max - self.y_min)
    
    def area(self) -> int:
        """计算面积"""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class ZLineDetector:
    """
    Z线检测器
    使用LoG (Laplacian of Gaussian) 或 DoG (Difference of Gaussian) 检测Z线
    """
    
    def __init__(self,
                 method: str = "log",
                 min_sigma: float = 1.0,
                 max_sigma: float = 4.0,
                 num_sigma: int = 10,
                 threshold: float = 0.1,
                 overlap: float = 0.5):
        """
        参数:
            method: 检测方法 "log" 或 "dog"
            min_sigma: 最小高斯核标准差
            max_sigma: 最大高斯核标准差
            num_sigma: sigma的数量
            threshold: 检测阈值（相对于最大响应）
            overlap: 允许的blob重叠比例
        """
        self.method = method
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        检测Z线位置
        
        参数:
            image: 输入的actinin通道图像
            
        返回:
            np.ndarray: shape (N, 2)，每行是一个Z线的 (y, x) 坐标
        """
        # 归一化图像到 [0, 1]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # 选择检测方法
        if self.method == "log":
            blobs = blob_log(
                image,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                threshold=self.threshold,
                overlap=self.overlap
            )
        elif self.method == "dog":
            blobs = blob_dog(
                image,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                threshold=self.threshold,
                overlap=self.overlap
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # blobs 格式: (y, x, sigma)，我们只需要坐标
        if len(blobs) == 0:
            return np.array([]).reshape(0, 2)
        
        coordinates = blobs[:, :2]  # (N, 2) - (y, x) 格式
        
        return coordinates
    
    def detect_multiscale(self, image: np.ndarray) -> np.ndarray:
        """
        多尺度检测，增加召回率
        
        参数:
            image: 输入图像
            
        返回:
            np.ndarray: 合并后的Z线坐标
        """
        all_coords = []
        
        # 使用多组参数
        param_sets = [
            {'min_sigma': 1, 'max_sigma': 3, 'threshold': 0.08},
            {'min_sigma': 2, 'max_sigma': 5, 'threshold': 0.1},
            {'min_sigma': 3, 'max_sigma': 6, 'threshold': 0.12},
        ]
        
        for params in param_sets:
            detector = ZLineDetector(
                method=self.method,
                min_sigma=params['min_sigma'],
                max_sigma=params['max_sigma'],
                threshold=params['threshold']
            )
            coords = detector.detect(image)
            if len(coords) > 0:
                all_coords.append(coords)
        
        if len(all_coords) == 0:
            return np.array([]).reshape(0, 2)
        
        # 合并并去重（使用简单的距离阈值）
        all_coords = np.vstack(all_coords)
        unique_coords = self._remove_duplicates(all_coords, min_dist=3.0)
        
        return unique_coords
    
    def _remove_duplicates(self, coords: np.ndarray, min_dist: float) -> np.ndarray:
        """移除距离过近的重复点"""
        if len(coords) <= 1:
            return coords
        
        from scipy.spatial import cKDTree
        
        tree = cKDTree(coords)
        keep_mask = np.ones(len(coords), dtype=bool)
        
        for i in range(len(coords)):
            if not keep_mask[i]:
                continue
            # 找到距离小于min_dist的邻居
            neighbors = tree.query_ball_point(coords[i], min_dist)
            for j in neighbors:
                if j > i:  # 只标记后面的点
                    keep_mask[j] = False
        
        return coords[keep_mask]


class SarcGraphPromptGenerator:
    """
    SarcGraph驱动的SAM提示框生成器
    
    工作流程:
    1. Z线检测 (LoG/DoG)
    2. DBSCAN聚类
    3. 边界框生成 + Padding
    """
    
    def __init__(self,
                 pixel_size_um: float = 0.5,
                 sarcomere_length_um: float = 2.0,
                 eps_factor: float = 2.0,
                 min_samples: int = 15,
                 padding_pixels: int = 20,
                 padding_ratio: float = 0.15,
                 use_hdbscan: bool = False,
                 min_cluster_size: int = 20):
        """
        参数:
            pixel_size_um: 每像素对应的微米数
            sarcomere_length_um: 肌节长度（微米），用于计算eps
            eps_factor: eps = sarcomere_length_um * eps_factor / pixel_size_um
            min_samples: DBSCAN的最小样本数
            padding_pixels: 边界框的固定padding（像素）
            padding_ratio: 边界框的比例padding
            use_hdbscan: 是否使用HDBSCAN（自动选择eps）
            min_cluster_size: HDBSCAN的最小簇大小
        """
        self.pixel_size_um = pixel_size_um
        self.sarcomere_length_um = sarcomere_length_um
        self.eps_factor = eps_factor
        self.min_samples = min_samples
        self.padding_pixels = padding_pixels
        self.padding_ratio = padding_ratio
        self.use_hdbscan = use_hdbscan
        self.min_cluster_size = min_cluster_size
        
        # 计算eps（像素单位）
        self.eps_pixels = (sarcomere_length_um * eps_factor) / pixel_size_um
        
        # Z线检测器
        self.zline_detector = ZLineDetector()
    
    def generate_prompts(self, 
                         actinin_image: np.ndarray,
                         return_details: bool = False) -> List[BoundingBox]:
        """
        生成SAM提示框
        
        参数:
            actinin_image: α-actinin通道图像
            return_details: 是否返回详细信息
            
        返回:
            List[BoundingBox]: 边界框列表
        """
        H, W = actinin_image.shape[:2]
        
        # 1. Z线检测
        zline_coords = self.zline_detector.detect_multiscale(actinin_image)
        
        if len(zline_coords) < self.min_samples:
            warnings.warn(f"检测到的Z线数量过少 ({len(zline_coords)})，可能是图像质量问题")
            return []
        
        # 注意：blob检测返回的是 (y, x) 格式，需要转换为 (x, y)
        zline_coords_xy = zline_coords[:, ::-1]  # (y, x) -> (x, y)
        
        # 2. DBSCAN/HDBSCAN聚类
        if self.use_hdbscan:
            clustering = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            ).fit(zline_coords_xy)
        else:
            clustering = DBSCAN(
                eps=self.eps_pixels,
                min_samples=self.min_samples
            ).fit(zline_coords_xy)
        
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声标签
        
        # 3. 为每个簇生成边界框
        boxes = []
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_points = zline_coords_xy[cluster_mask]
            
            # 计算边界
            x_min = int(cluster_points[:, 0].min())
            x_max = int(cluster_points[:, 0].max())
            y_min = int(cluster_points[:, 1].min())
            y_max = int(cluster_points[:, 1].max())
            
            # 计算padding
            width = x_max - x_min
            height = y_max - y_min
            
            pad_x = max(self.padding_pixels, int(width * self.padding_ratio))
            pad_y = max(self.padding_pixels, int(height * self.padding_ratio))
            
            # 应用padding并确保不超出图像边界
            x_min = max(0, x_min - pad_x)
            x_max = min(W, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(H, y_max + pad_y)
            
            # 创建边界框
            box = BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=1.0,
                num_zlines=int(cluster_mask.sum()),
                cluster_id=int(cluster_id)
            )
            boxes.append(box)
        
        return boxes
    
    def generate_prompts_with_stats(self, 
                                    actinin_image: np.ndarray) -> Dict:
        """
        生成提示框并返回详细统计信息
        
        返回:
            dict: 包含boxes, zline_coords, labels等信息
        """
        H, W = actinin_image.shape[:2]
        
        # Z线检测
        zline_coords = self.zline_detector.detect_multiscale(actinin_image)
        zline_coords_xy = zline_coords[:, ::-1] if len(zline_coords) > 0 else np.array([]).reshape(0, 2)
        
        # 聚类
        if len(zline_coords_xy) >= self.min_samples:
            if self.use_hdbscan:
                clustering = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples
                ).fit(zline_coords_xy)
            else:
                clustering = DBSCAN(
                    eps=self.eps_pixels,
                    min_samples=self.min_samples
                ).fit(zline_coords_xy)
            labels = clustering.labels_
        else:
            labels = np.array([])
        
        # 生成boxes
        boxes = self.generate_prompts(actinin_image) if len(zline_coords) >= self.min_samples else []
        
        return {
            'boxes': boxes,
            'zline_coords': zline_coords_xy,
            'labels': labels,
            'num_zlines': len(zline_coords),
            'num_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'num_noise_points': int((labels == -1).sum()) if len(labels) > 0 else 0,
            'image_shape': (H, W),
            'eps_pixels': self.eps_pixels
        }
    
    def boxes_to_tensor(self, 
                        boxes: List[BoundingBox], 
                        device: str = 'cpu') -> 'torch.Tensor':
        """
        将边界框列表转换为PyTorch tensor
        
        返回:
            torch.Tensor: shape (N, 4)，格式为 [x1, y1, x2, y2]
        """
        import torch
        
        if len(boxes) == 0:
            return torch.zeros((0, 4), device=device)
        
        box_array = np.array([box.to_xyxy() for box in boxes])
        return torch.tensor(box_array, dtype=torch.float32, device=device)


class AdaptivePromptGenerator(SarcGraphPromptGenerator):
    """
    自适应提示框生成器
    根据图像特性自动调整参数
    """
    
    def __init__(self, pixel_size_um: float = 0.5):
        super().__init__(pixel_size_um=pixel_size_um)
    
    def _estimate_optimal_eps(self, zline_coords: np.ndarray) -> float:
        """
        使用k-distance图自动估计最佳eps
        """
        from sklearn.neighbors import NearestNeighbors
        
        if len(zline_coords) < 10:
            return self.eps_pixels
        
        # 计算每个点到第k个最近邻的距离
        k = min(self.min_samples, len(zline_coords) - 1)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(zline_coords)
        distances, _ = neigh.kneighbors(zline_coords)
        
        # 对k-distance排序
        k_distances = np.sort(distances[:, -1])
        
        # 找到"肘部"（曲率最大的点）
        # 简单方法：使用梯度变化
        gradients = np.gradient(k_distances)
        elbow_idx = np.argmax(gradients)
        
        optimal_eps = k_distances[elbow_idx]
        
        # 限制在合理范围内
        optimal_eps = np.clip(optimal_eps, self.eps_pixels * 0.5, self.eps_pixels * 2.0)
        
        return optimal_eps
    
    def generate_prompts(self, 
                         actinin_image: np.ndarray,
                         return_details: bool = False) -> List[BoundingBox]:
        """自适应生成提示框"""
        
        # Z线检测
        zline_coords = self.zline_detector.detect_multiscale(actinin_image)
        
        if len(zline_coords) < self.min_samples:
            return []
        
        zline_coords_xy = zline_coords[:, ::-1]
        
        # 自适应估计eps
        optimal_eps = self._estimate_optimal_eps(zline_coords_xy)
        
        # 使用估计的eps进行聚类
        clustering = DBSCAN(
            eps=optimal_eps,
            min_samples=self.min_samples
        ).fit(zline_coords_xy)
        
        # 后续与父类相同
        self.eps_pixels = optimal_eps  # 临时更新
        boxes = super().generate_prompts(actinin_image, return_details)
        
        return boxes


# ============ 测试代码 ============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 创建模拟的actinin图像
    H, W = 512, 512
    image = np.zeros((H, W), dtype=np.float32)
    
    # 添加模拟的心肌细胞（带有条纹纹理）
    # 细胞1：位置 (100, 100) 到 (250, 200)
    for i in range(100, 250, 8):  # 条纹间隔约8像素
        image[100:200, i:i+3] = np.random.uniform(0.7, 1.0)
    
    # 细胞2：位置 (300, 150) 到 (450, 280)
    for i in range(300, 450, 8):
        image[150:280, i:i+3] = np.random.uniform(0.7, 1.0)
    
    # 添加背景噪声
    image += np.random.uniform(0, 0.1, (H, W))
    
    # 创建提示框生成器
    generator = SarcGraphPromptGenerator(
        pixel_size_um=0.5,
        sarcomere_length_um=2.0,
        eps_factor=2.0,
        min_samples=10,
        padding_pixels=15
    )
    
    # 生成提示框
    result = generator.generate_prompts_with_stats(image)
    
    print("=" * 60)
    print("SarcGraph 提示框生成测试")
    print("=" * 60)
    print(f"图像大小: {result['image_shape']}")
    print(f"检测到的Z线数量: {result['num_zlines']}")
    print(f"聚类数量（细胞数）: {result['num_clusters']}")
    print(f"噪声点数量: {result['num_noise_points']}")
    print(f"使用的eps (像素): {result['eps_pixels']:.2f}")
    print()
    
    for i, box in enumerate(result['boxes']):
        print(f"细胞 {i+1}:")
        print(f"  边界框: {box.to_xyxy()}")
        print(f"  Z线数量: {box.num_zlines}")
        print(f"  面积: {box.area()} 像素²")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
