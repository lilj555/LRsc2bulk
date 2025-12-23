"""
GPU资源管理工具
"""
import torch
import psutil
import logging
from typing import List, Tuple, Optional
import subprocess

logger = logging.getLogger(__name__)

class GPUManager:
    """GPU资源管理器"""
    
    def __init__(self):
        self.device = None
        self.available_gpus = []
        self.memory_info = {}
        
    def detect_gpus(self) -> List[int]:
        """检测可用GPU"""
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU")
            return []
        
        gpu_count = torch.cuda.device_count()
        available_gpus = []
        supported_arch = set()
        try:
            arch_list = torch.cuda.get_arch_list()
            supported_arch = set(arch_list)
        except Exception:
            supported_arch = set()
        
        for i in range(gpu_count):
            try:
                # 检查GPU内存
                torch.cuda.set_device(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_free = memory_total - memory_allocated
                cap = torch.cuda.get_device_capability(i)
                arch_tag = f"sm_{cap[0]}{cap[1]}"
                
                self.memory_info[i] = {
                    'total': memory_total / 1024**3,  # GB
                    'allocated': memory_allocated / 1024**3,
                    'free': memory_free / 1024**3,
                    'arch': arch_tag
                }
                
                # 过滤不受当前PyTorch构建支持的架构（例如 Blackwell sm_120）
                if supported_arch and arch_tag not in supported_arch:
                    logger.warning(
                        f"GPU {i} ({torch.cuda.get_device_name(i)}) 架构 {arch_tag} "
                        f"不被当前PyTorch({torch.__version__})支持，跳过该设备"
                    )
                    continue
                
                # 如果有足够的空闲内存（至少2GB），认为GPU可用
                if memory_free > 2 * 1024**3:
                    available_gpus.append(i)
                    logger.info(
                        f"GPU {i}: {torch.cuda.get_device_name(i)}, "
                        f"架构: {arch_tag}, 内存: {self.memory_info[i]['free']:.1f}GB 可用"
                    )
                
            except Exception as e:
                logger.warning(f"检测GPU {i}时出错: {e}")
        
        self.available_gpus = available_gpus
        return available_gpus
    
    def select_best_gpu(self, min_memory_gb: float = 2.0) -> Optional[int]:
        """选择最佳GPU"""
        if not self.available_gpus:
            return None
        
        # 选择空闲内存最多的GPU
        best_gpu = max(self.available_gpus, 
                      key=lambda x: self.memory_info[x]['free'])
        
        if self.memory_info[best_gpu]['free'] >= min_memory_gb:
            return best_gpu
        else:
            logger.warning(f"最佳GPU {best_gpu}的可用内存"
                         f"({self.memory_info[best_gpu]['free']:.1f}GB)"
                         f"小于要求的{min_memory_gb}GB")
            return None
    
    def setup_device(self, device_ids: Optional[List[int]] = None, 
                    min_memory_gb: float = 2.0) -> torch.device:
        """设置计算设备"""
        available_gpus = self.detect_gpus()
        
        if not available_gpus:
            logger.info("使用CPU进行训练")
            self.device = torch.device('cpu')
            return self.device
        
        if device_ids:
            # 使用指定的GPU
            valid_gpus = [gpu for gpu in device_ids if gpu in available_gpus]
            if not valid_gpus:
                logger.warning("指定的GPU不可用，自动选择最佳GPU")
                best_gpu = self.select_best_gpu(min_memory_gb)
            else:
                best_gpu = valid_gpus[0]
        else:
            # 自动选择最佳GPU
            best_gpu = self.select_best_gpu(min_memory_gb)
        
        if best_gpu is not None:
            self.device = torch.device(f'cuda:{best_gpu}')
            torch.cuda.set_device(best_gpu)
            logger.info(f"使用GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)}")
        else:
            logger.info("GPU内存不足，使用CPU进行训练")
            self.device = torch.device('cpu')
        
        return self.device
    
    def get_memory_usage(self) -> dict:
        """获取当前内存使用情况"""
        if self.device and self.device.type == 'cuda':
            gpu_id = self.device.index
            return {
                'allocated': torch.cuda.memory_allocated(gpu_id) / 1024**3,
                'cached': torch.cuda.memory_reserved(gpu_id) / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated(gpu_id) / 1024**3
            }
        return {}
    
    def get_gpu_utilization(self) -> Optional[dict]:
        """获取GPU利用率和显存使用（需要nvidia-smi）"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            util = {}
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                idx = int(parts[0])
                name = parts[1]
                gpu_util = float(parts[2])
                mem_used = float(parts[3]) / 1024.0
                mem_total = float(parts[4]) / 1024.0
                util[idx] = {
                    "name": name,
                    "utilization_percent": gpu_util,
                    "memory_used_gb": mem_used,
                    "memory_total_gb": mem_total
                }
            return util
        except Exception:
            return None
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")

def get_optimal_batch_size(model, sample_input, device, 
                          max_batch_size: int = 512) -> int:
    """自动确定最优批次大小"""
    if device.type == 'cpu':
        # CPU情况下，根据内存大小确定批次大小
        memory_gb = psutil.virtual_memory().total / 1024**3
        if memory_gb >= 32:
            return min(128, max_batch_size)
        elif memory_gb >= 16:
            return min(64, max_batch_size)
        else:
            return min(32, max_batch_size)
    
    # GPU情况下，通过试验确定最大批次大小
    model.eval()
    batch_size = 2
    
    while batch_size <= max_batch_size:
        try:
            # 创建测试批次
            if isinstance(sample_input, tuple):
                test_input = tuple(x[:batch_size].to(device) if hasattr(x, 'to') 
                                 else x for x in sample_input)
            else:
                test_input = sample_input[:batch_size].to(device)
            
            # 前向传播测试
            with torch.no_grad():
                _ = model(test_input)
            
            torch.cuda.empty_cache()
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                optimal_batch_size = max(1, batch_size // 4)
                logger.info(f"自动确定最优批次大小: {optimal_batch_size}")
                return optimal_batch_size
            else:
                raise e
    
    return min(batch_size // 2, max_batch_size)

def setup_mixed_precision() -> Tuple[bool, Optional[torch.cuda.amp.GradScaler]]:
    """设置混合精度训练"""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        logger.info("启用混合精度训练")
        return True, torch.cuda.amp.GradScaler()
    else:
        logger.info("不支持混合精度训练或GPU算力不足")
        return False, None
