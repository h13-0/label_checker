import cv2
import numpy as np
from readerwriterlock import rwlock


class HSVFilter():
    def __init__(self) -> None:
        self._h_low = 0
        self._h_high = 255
        self._s_low = 0
        self._s_high = 255
        self._v_low = 0
        self._v_high = 255
        self._rw_lock = rwlock.RWLockFairD()

    @property
    def h_low(self):
        with self._rw_lock.gen_rlock():
            return self._h_low
    
    @h_low.setter
    def h_low(self, h_low):
        with self._rw_lock.gen_wlock():
            self._h_low = h_low
    
    @property
    def h_high(self):
        with self._rw_lock.gen_rlock():
            return self._h_high
    
    @h_high.setter
    def h_high(self, h_high):
        with self._rw_lock.gen_wlock():
            self._h_high = h_high

    @property
    def s_low(self):
        with self._rw_lock.gen_rlock():
            return self._s_low
    
    @s_low.setter
    def s_low(self, s_low):
        with self._rw_lock.gen_wlock():
            self._s_low = s_low

    @property
    def s_high(self):
        with self._rw_lock.gen_rlock():
            return self._s_high
    
    @s_high.setter
    def s_high(self, s_high):
        with self._rw_lock.gen_wlock():
            self._s_high = s_high
    
    @property
    def v_low(self):
        with self._rw_lock.gen_rlock():
            return self._v_low
    
    @v_low.setter
    def v_low(self, v_low):
        with self._rw_lock.gen_wlock():
            self._v_low = v_low

    @property
    def v_high(self):
        with self._rw_lock.gen_rlock():
            return self._v_high
    
    @v_high.setter
    def v_high(self, v_high):
        with self._rw_lock.gen_wlock():
            self._v_high = v_high

    def to_hsv_full(self, input:np.ndarray):
        with self._rw_lock.gen_rlock():
            return cv2.cvtColor(input, cv2.COLOR_BGR2HSV_FULL)


    def filter(self, input:np.ndarray):
        """
        @brief: 将输入图像按照选定HSV阈值进行过滤
        @return: HSV阈值外的部分置黑, 阈值内的图像保留原色
        """
        hsv_img = self.to_hsv_full(input)
        mask = None
        with self._rw_lock.gen_rlock():
            mask = cv2.inRange(
                hsv_img, 
                np.array([self._h_low, self._s_low, self._v_low]), 
                np.array([self._h_high, self._s_high, self._v_high])
            )
        B, G, R = cv2.split(input)
        return cv2.merge([
            cv2.bitwise_and(B, mask),
            cv2.bitwise_and(G, mask),
            cv2.bitwise_and(R, mask),
        ])

