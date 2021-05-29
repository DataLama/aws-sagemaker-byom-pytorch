import os
import torch
import numpy as np
from os import environ
from psutil import cpu_count
from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, BatchEncoding
from transformers.convert_graph_to_onnx import convert, quantize

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions

@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)

@dataclass
class OnnxInferenceResult:
    model_inference_time: [int]  
    optimized_model_path: str
        
class InferenceTimeBenchTester:
    def __init__(self, model_path, tokenizer_path, provider='CPUExecutionProvider', framework='onnx', use_qt=False):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.provider = provider
        self.framework = framework
        self.device = "cpu" if self.provider=='CPUExecutionProvider' else "cuda:0"
        
        if framework == 'pt':
            config = AutoConfig.from_pretrained(os.path.join(model_path, 'config.json'))
            if 'nsmc' in model_path:
                model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, 'pytorch_model.bin'), config=config).to(self.device).eval()
            else:
                model = AutoModelForTokenClassification.from_pretrained(os.path.join(model_path, 'pytorch_model.bin'), config=config).to(self.device).eval()
            if use_qt:
                quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                self.model = quantized_model
            else:
                self.model = model
        elif framework == 'onnx':
            if use_qt:
                quantized_model_path = quantize(Path(model_path))
                self.model = self.create_model_for_provider(quantized_model_path.as_posix(), provider)
            else:
                self.model = self.create_model_for_provider(model_path, provider)
        else:
            raise ValueError('No model is loaded')
        
    
    def create_model_for_provider(self, model_path: str, provider: str) -> InferenceSession: 

        assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend 
        session = InferenceSession(model_path, options, providers=[provider])
        session.disable_fallback()

        return session
    
    def run(self):
        test_case = self.inputs
        time_buffer = []
        n = 1000
        
        if self.framework == 'pt':
            # Warm up the overall model to have a fair comparaison
            for _ in range(10):
                with torch.no_grad():
                    outputs = self.model(**test_case)

            for _ in trange(n, desc=f"Tracking inference time on {self.provider} with {self.model_path}"):
                with track_infer_time(time_buffer):
                    with torch.no_grad():
                        outputs = self.model(**test_case)
            
        
        elif self.framework == 'onnx':
            # Warm up the overall model to have a fair comparaison
            for _ in range(10):
                outputs = self.model.run(None, test_case)

            for _ in trange(n, desc=f"Tracking inference time on {self.provider} with {self.model_path}"):
                with track_infer_time(time_buffer):
                    outputs = self.model.run(None, test_case)

        return OnnxInferenceResult(
                time_buffer, 
                None
            )
    
    @property
    def inputs(self):
        doc = """네이버(NAVER)가 국내 기업 최초로 초대규모(Hyper scale) 인공지능(AI)인 '하이퍼클로바'(HyperCLOVA)를 공개했다.
                네이버는 25일 온라인으로 '네이버 AI NOW' 콘퍼런스를 열고 국내 AI 퍼스트무버를 넘어 글로벌 AI 기술 리더로 발돋움하겠다는 계획을 밝혔다.
                하이퍼클로바는 기존 한국어 AI 패러다임을 바꿔 이용자를 비롯해 중소사업자(SME), 창작자에게 차별화된 경험을 선보인다는 계획이다. '모두를 위한 AI' 시대를 여는 것이 네이버의 포부다.
                네이버가 국내 기업 최초로 자체 개발한 초대규모 AI다. 슈퍼컴퓨터를 이용해 확장된 매개변수를 바탕으로 다른 AI 모델 개발의 기본을 마련한다.
                정석근 네이버 클로바(CLOVA) CIC 대표는 "글로벌 기술 대기업은 대형 AI 모델이 가져올 파괴적 혁신에 대한 기대로 투자를 가속화하고 있다"며 "한국 AI 기술이 글로벌 플랫폼에 종속되지 않으려면 이미 공개된 기술을 활용하고 따라잡는 수준에 그칠 수 없다고 판단했다"고 말했다.
                하이퍼클로바는 지난해 미국에서 공개된 오픈AI의 'GPT-3'(175B)를 뛰어넘는 204B(2040억개) 파라미터(매개변수) 규모로 개발됐다. AI 모델의 크기를 나타내는 파라미터 수가 많아질수록 더욱 많은 문제를 해결할 수 있다.
                하이퍼클로바는 GPT-3보다 한국어 데이터를 6500배 이상 학습한 세계에서 가장 큰(현재) 한국어 초거대 언어모델이기도 하다. 영어가 학습 데이터 대부분을 차지하는 GPT-3와 달리 하이퍼클로바 학습 데이터는 한국어 비중이 97%에 달한다. 영어 중심 글로벌 AI 모델과 달리 한국어에 최적화한 언어모델을 개발함으로써 AI 주권을 확보한다는 의미도 있다.
                네이버는 지난 10월 국내 기업 최초로 700페타플롭(PF) 성능의 슈퍼컴퓨터를 도입하며 대용량 데이터 처리를 위한 인프라를 갖췄다.
                국내 최대 인터넷 플랫폼을 운영하며 쌓아온 대규모 데이터 처리 능력도 하이퍼클로바만의 중요한 경쟁력이다. 네이버는 하이퍼클로바 개발을 위해 5600억개 토큰의 한국어 대용량 데이터를 구축했다.
                슈퍼컴퓨터 인프라와 한국어 데이터 외에 네이버가 보유한 세계 최고 수준의 AI 연구 개발 역량 역시 하이퍼클로바 자체 개발에 영향을 활용됐다. 네이버는 지난 한해 동안 글로벌 톱 AI 콘퍼런스에서 국내 기업 중 가장 많은 43개의 정규 논문을 발표하며 기술력을 인정받았다.
                서울대와 '서울대-네이버 초대규모(Hyperscale) AI 연구센터'를 설립하고, 카이스트 AI 대학원과는 '카이스트-네이버 초창의적(Hypercreative) AI 연구센터'를 설립하는 등 긴밀하고 강력한 산학협력을 통해 AI 공동 연구에도 힘쓴다.
                네이버는 한국어 외 다른 언어로 언어 모델을 확장하고 언어뿐만 아니라 영상이나 이미지 등도 이해하는 '멀티모달(Multimodal) AI'로 하이퍼클로바를 계속해서 발전시켜나갈 계획이다.
                정 대표는 "더 짧은 시간과 더 적은 리소스를 사용해 이전에 우리가 상상만 했던, 또는 우리가 상상하지 못했던 일들마저 가능해지는 새로운 AI 시대가 열리고 있다"면서 "하이퍼클로바를 통해 SME와 크리에이터를 포함해 AI 기술이 필요한 모두에게 새로운 경험을 제공할 것"이라고 말했다."""
        
        if self.framework == 'pt':
            model_inputs = self.tokenizer(doc, return_tensors='pt' , truncation=True)
            model_inputs_on_device = {arg_name: tensor.to(self.device) for arg_name, tensor in model_inputs.items()}
            return model_inputs_on_device
            
        elif self.framework == 'onnx':
            model_inputs = self.tokenizer(doc, return_tensors='pt' , truncation=True)
            inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
            return inputs_onnx
        else:
            raise ValueError('Choose framework "pt" or "onnx".')

