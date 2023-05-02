from EISeg.eiseg.controller import InteractiveController
import numpy as np
import cv2

class Segmenter:
    def __init__(self, params_path):
        predictor_params = {
            "brs_mode": "NoBRS",
            "with_flip": False,
            "zoom_in_params": {
                "skip_clicks": -1,
                "target_size": (400, 400),
                "expansion_ratio": 1.4,
            },
            "predictor_params": {
                "net_clicks_limit": None,
                "max_size": 800,
                "with_mask": True,
            },
        }
        self.controller = InteractiveController(predictor_params=predictor_params)
        self.controller.setModel(params_path)
        self.controller.addLabel(0, "", (0, 0, 0))

    def segment(self, image, positive_points, negative_points=[]):
        image = np.array(image * 255.).astype('uint8')
        self.controller.setImage(image)
        self.controller.resetLastObject()
        for point in positive_points[:-1]:
            self.controller.addClick(point[0], point[1], True, False)
        for point in negative_points:
            self.controller.addClick(point[0], point[1], False, False)
        self.controller.addClick(positive_points[-1][0], positive_points[-1][1], True, True)
        return self.controller.probs_history[0][1]


# segmenter = Segmenter("DS_NeRF/EISeg/pretrained_weights/static_edgeflow_cocolvis.pdiparams")
# segmenter = Segmenter("DS_NeRF/EISeg/pretrained_weights/static_hrnet18s_ocr48_cocolvis.pdiparams")