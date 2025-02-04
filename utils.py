import torch
import gc

def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_info():
    """GPU 정보 출력"""
    if torch.cuda.is_available():
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"현재 GPU: {torch.cuda.current_device()}")
    else:
        print("사용 가능한 GPU가 없습니다.") 