import cv2
import numpy as np


def inverse_crf(Z, gamma=2.2):
    """ 应用逆CRF（伽马校正的逆函数） """
    return np.power(Z, gamma)


def apply_inverse_crf_and_normalize(image_path, exposure_time, gamma=2.2):
    """ 读取图像，应用逆CRF，并除以曝光时间 """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded, check the path.")

    # 将图像数据转换为0到1的范围
    img = img.astype(np.float32) / 255.0

    # 应用逆CRF
    img_linear = inverse_crf(img, gamma)

    # 除以曝光时间
    img_corrected = img_linear / exposure_time

    # 将结果重新缩放到0-255并转换为整数
    # img_corrected = np.clip(img_corrected * 255.0, 0, 255).astype(np.uint8)

    return img_corrected


def transfer_hdr_to_ldr(hdr_image, exposure_time, gamma=2.2):
    """ 将HDR图像转换为LDR图像 """

    ldr_image = np.clip(np.power((hdr_image * exposure_time), 1 / gamma), 0, 1)
    return ldr_image


if __name__ == '__main__':
    # 使用示例
    image_path = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken Left Images/000000.png'
    exposure_time = 0.034  # 曝光时间，根据需要调整
    gamma_value = 2.2  # 伽马值，根据CRF调整

    # 处理图像
    hdr_image = apply_inverse_crf_and_normalize(image_path, exposure_time, gamma_value)
    result_image = transfer_hdr_to_ldr(hdr_image, exposure_time / 3, gamma_value)
    print(result_image)

    # 保存或显示结果
    cv2.imshow('Processed Image', result_image)
    print('press \'q\' or \'Esc\' to quit')
    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):  # 按下 ESC(27) 或 'q' 退出
        cv2.destroyAllWindows()
