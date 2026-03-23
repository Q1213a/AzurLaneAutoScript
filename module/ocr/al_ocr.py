import os
import numpy as np
import cv2
from PIL import Image

from module.exception import RequestHumanTakeover
from module.logger import logger

class AlOcr:
    # 设为 True 可将每张送入 OCR 的图片保存到 debug_ocr/ 目录以便调试
    DEBUG = False

    def __init__(self, **kwargs):
        self.model = None
        self.name = kwargs.get('name', 'en')
        self.params = {}
        self.is_pdparams = False
        self._model_loaded = False
        import os
        logger.info(f"[VERBOSE] Created AlOcr instance: name='{self.name}', kwargs={kwargs}, PID={os.getpid()}")
        
    def _read_gpu_acceleration_setting(self) -> bool:
        """
        Read UseOcrGpuAcceleration from the current instance's config JSON.
        Uses ALAS_CONFIG_NAME env var (set by process_manager) to locate the file.
        Defaults to True if the config cannot be read.
        """
        import json
        config_name = os.environ.get('ALAS_CONFIG_NAME', '')
        if not config_name:
            return True
        path = f'config/{config_name}.json'
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            val = data.get('Alas', {}).get('Optimization', {}).get('UseOcrGpuAcceleration', True)
            if val is False:
                logger.info(f'OCR GPU acceleration disabled by {path}')
                return False
        except Exception:
            pass
        return True

    def init(self):
        import time
        import os
        start_time = time.time()
        logger.info(f"[VERBOSE] AlOcr.init() started for '{self.name}'. Process ID: {os.getpid()}")
        try:
            logger.info('[VERBOSE] 正在加载 OCR 模型...')
            from rapidocr import RapidOCR, OCRVersion
            logger.info('[VERBOSE] 成功 load rapidocr: RapidOCR, OCRVersion')
        except Exception as e:
            logger.critical(f'Failed to load OCR dependencies: {e}')
            logger.critical(f'无法加载 OCR 依赖，如错误信息包含 DLL load failed while 请安装微软 C++ 运行库 https://aka.ms/vs/17/release/vc_redist.x64.exe')
            return

        if self.name == 'zhcn' or self.name == 'cn':
            self.params = {
                "Global.use_det": False,
                "Global.use_cls": False,
                "Det.model_path": None,
                "Cls.model_path": None,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_path": "bin/ocr_models/zh-CN/alocr-zh-cn-v2.5.dtk.onnx",
                "Rec.rec_keys_path": "bin/ocr_models/zh-CN/cn.txt",
                "EngineConfig.onnxruntime.use_dml": True
            }
        else:
            self.params = {
                "Global.use_det": False,
                "Global.use_cls": False,
                "Det.model_path": None,
                "Cls.model_path": None,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
                "Rec.model_path": "bin/ocr_models/en-US/alocr-en-us-v2.0.nvc.onnx",
                "Rec.rec_keys_path": "bin/ocr_models/en-US/en.txt",
                "EngineConfig.onnxruntime.use_dml": True
            }

        use_gpu = self._read_gpu_acceleration_setting()
        logger.info(f"[VERBOSE] GPU acceleration setting read: {use_gpu}")
        if use_gpu:
            self.params['EngineConfig.onnxruntime.use_dml'] = True
        else:
            self.params.pop('EngineConfig.onnxruntime.use_dml', None)
        
        # ----- Direct pdparams support for testing -----
        model_path = self.params.get("Rec.model_path", "")
        self.is_pdparams = model_path.endswith('.pdparams')
        logger.info(f"[VERBOSE] Model path: {model_path}, is_pdparams: {self.is_pdparams}")
        if self.is_pdparams:
            logger.info("[VERBOSE] Detected pdparams checkpoint. Preparing native PaddleOCR inference...")
            import sys
            paddleocr_path = r"c:\Users\AzurLane\Desktop\Projects\AzurLaneAutoScript\PaddleOCR"
            if paddleocr_path not in sys.path:
                sys.path.append(paddleocr_path)
            self._init_pdparams_model(model_path)
        else:
            logger.info(f"[VERBOSE] Calling kwargs RapidOCR initialized with: {self.params}")
            self.model = RapidOCR(params=self.params)
            
        logger.info(f"OCR model '{self.name}' Initialized. pdparams: {self.is_pdparams}, params: {self.params}, took: {time.time() - start_time:.4f}s")
        self._model_loaded = True

    def unload(self):
        """卸载并清理加载的 OCR 模型以释放内存"""
        if self._model_loaded:
            logger.info(f"Unloading OCR model '{self.name}' from memory...")
            self.model = None
            if hasattr(self, 'post_process_class'):
                del self.post_process_class
            self._model_loaded = False
            import gc
            gc.collect()

    def _init_pdparams_model(self, pdparams_path):
        import yaml
        import os
        import paddle
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        
        # 使用 module/ocr/ 下的本地 yaml 配置
        ocr_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(ocr_dir, '..', '..'))
        
        if "en-US" in pdparams_path:
            yaml_path = os.path.join(ocr_dir, 'en.yaml')
        else:
            yaml_path = os.path.join(ocr_dir, 'cn.yaml')
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 将字典的相对路径转为绝对路径（相对于项目根目录）
        dict_path = config['Global'].get('character_dict_path', '')
        if dict_path and not os.path.isabs(dict_path):
            config['Global']['character_dict_path'] = os.path.join(project_root, dict_path)
            
        global_config = config['Global']
        global_config['infer_mode'] = True
        config['Architecture']['Transform'] = None
        
        # Build PostProcess
        self.post_process_class = build_post_process(config['PostProcess'], global_config)
        
        # Inject MultiHead out_channels_list
        if hasattr(self.post_process_class, "character"):
            char_num = len(getattr(self.post_process_class, "character"))
            if config["Architecture"]["Head"]["name"] == "MultiHead":
                out_channels_list = {}
                out_channels_list["CTCLabelDecode"] = char_num
                out_channels_list["SARLabelDecode"] = char_num + 2
                out_channels_list["NRTRLabelDecode"] = char_num + 3
                config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
            else:
                config["Architecture"]["Head"]["out_channels"] = char_num
                
        # Build Model
        self.model = build_model(config['Architecture'])
        
        # Load params
        state_dict = paddle.load(pdparams_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()

    def _ensure_loaded(self):
        if not self._model_loaded:
            self.init()

    def _preprocess_pdparams(self, img):
        import math
        imgH, imgW = 48, 320
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im[np.newaxis, :]

    def _infer_pdparams(self, img):
        import paddle
        import time
        start_t = time.time()
        logger.info(f"[VERBOSE] AlOcr._infer_pdparams: pdparams inferencing on image with shape {img.shape if hasattr(img, 'shape') else 'unknown'}, dtype: {img.dtype if hasattr(img, 'dtype') else 'unknown'}")
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            tensor_data = self._preprocess_pdparams(img)
            logger.info(f"[VERBOSE] AlOcr._infer_pdparams: processed tensor_data shape {tensor_data.shape}, min {tensor_data.min()}, max {tensor_data.max()}")
            tensor = paddle.to_tensor(tensor_data)
            logger.info(f"[VERBOSE] AlOcr._infer_pdparams: Executing model(tensor)...")
            preds = self.model(tensor)
            logger.info(f"[VERBOSE] AlOcr._infer_pdparams: Model execution returned, passing to post_process_class...")
            post_result = self.post_process_class(preds)
            logger.info(f"[VERBOSE] AlOcr._infer_pdparams: post_result raw: {post_result}")
            if post_result and post_result[0]:
                res = post_result[0][0]
                logger.info(f"[VERBOSE] AlOcr._infer_pdparams result: {res}, took: {time.time() - start_t:.4f}s")
                return res
            logger.info(f"[VERBOSE] AlOcr._infer_pdparams result: <empty>, took: {time.time() - start_t:.4f}s")
            return ""
        except Exception as e:
            logger.error(f"[VERBOSE] AlOcr._infer_pdparams exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def ocr(self, img_fp):
        import time
        start_t = time.time()
        logger.info(f"[VERBOSE] AlOcr.ocr: Ensure loaded...")
        self._ensure_loaded()
        if hasattr(img_fp, 'shape'):
            logger.info(f"AlOcr.ocr: inferencing on single image shape {img_fp.shape}, dtype {img_fp.dtype if hasattr(img_fp, 'dtype') else 'unknown'}")
        else:
            logger.info(f"AlOcr.ocr: inferencing on {type(img_fp)}")
            
        if self.DEBUG and isinstance(img_fp, np.ndarray):
            os.makedirs('debug_ocr', exist_ok=True)
            debug_path = f'debug_ocr/{int(time.time() * 1000)}.png'
            logger.info(f"[VERBOSE] AlOcr.ocr: Saving debug image to {debug_path}")
            cv2.imwrite(debug_path, img_fp)
            
        try:
            if self.is_pdparams:
                res = self._infer_pdparams(img_fp)
                logger.info(f"[VERBOSE] AlOcr.ocr pdparams return: {res}")
                return res
            else:
                logger.info("[VERBOSE] AlOcr.ocr: Calling RapidOCR model()...")
                res = self.model(img_fp)
                logger.debug(f"[VERBOSE] AlOcr.ocr: RapidOCR model() returned raw: {res}")
                if hasattr(res, 'txts') and res.txts:
                    logger.info(f"AlOcr.ocr result: {res.txts[0]}, took: {time.time() - start_t:.4f}s")
                    return res.txts[0]
                logger.info(f"AlOcr.ocr result: <empty>, took: {time.time() - start_t:.4f}s")
                return ""
        except Exception as e:
            logger.error(f"[VERBOSE] AlOcr.ocr exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def ocr_for_single_line(self, img_fp):
        return self.ocr(img_fp)

    def ocr_for_single_lines(self, img_list):
        import time
        start_t = time.time()
        logger.info(f"[VERBOSE] AlOcr.ocr_for_single_lines: Ensure loaded...")
        self._ensure_loaded()
        logger.info(f"AlOcr.ocr_for_single_lines: batch inferencing on {len(img_list)} images")
        results = []
        for i, img in enumerate(img_list):
            if hasattr(img, 'shape'):
                logger.info(f"  [{i}/{len(img_list)}] image shape: {img.shape}, dtype: {img.dtype if hasattr(img, 'dtype') else 'unknown'}")
            if self.DEBUG and isinstance(img, np.ndarray):
                os.makedirs('debug_ocr', exist_ok=True)
                debug_path = f'debug_ocr/{int(time.time() * 1000)}_{id(img)}.png'
                logger.info(f"[VERBOSE] Saving batch debug image {i} to {debug_path}")
                cv2.imwrite(debug_path, img)
            
            try:
                if self.is_pdparams:
                    logger.info(f"[VERBOSE]   [{i}/{len(img_list)}] Calling _infer_pdparams...")
                    res = self._infer_pdparams(img)
                    logger.debug(f"[VERBOSE]   [{i}/{len(img_list)}] _infer_pdparams raw output: {res}")
                    results.append(res)
                else:
                    logger.info(f"[VERBOSE]   [{i}/{len(img_list)}] Calling RapidOCR model()...")
                    res = self.model(img)
                    logger.debug(f"[VERBOSE]   [{i}/{len(img_list)}] RapidOCR raw result: {res}")
                    if hasattr(res, 'txts') and res.txts:
                        results.append(res.txts[0])
                    else:
                        results.append("")
            except Exception as e:
                logger.error(f"[VERBOSE] AlOcr.ocr_for_single_lines exception on image {i}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results.append("")
        logger.info(f"AlOcr.ocr_for_single_lines results: {results}, took: {time.time() - start_t:.4f}s")
        return results

    def set_cand_alphabet(self, cand_alphabet):
        pass

    def atomic_ocr(self, img_fp, cand_alphabet=None):
        res = self.ocr(img_fp)
        if cand_alphabet:
            res = ''.join([c for c in res if c in cand_alphabet])
        return res

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        res = self.ocr_for_single_line(img_fp)
        if cand_alphabet:
            res = ''.join([c for c in res if c in cand_alphabet])
        return res

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None):
        results = self.ocr_for_single_lines(img_list)
        if cand_alphabet:
            results = [''.join([c for c in res if c in cand_alphabet]) for res in results]
        return results

    def debug(self, img_list):
        """
        Visual debugging of images fed to OCR.
        """
        if len(img_list) > 0:
            # Ensure images are properly formatted for hconcat
            concat_list = []
            for img in img_list:
                if len(img.shape) == 2:
                    # Gray to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                concat_list.append(img)
                
            image = cv2.hconcat(concat_list)
            Image.fromarray(image).show()
