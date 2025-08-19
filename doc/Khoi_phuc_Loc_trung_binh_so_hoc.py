import numpy as np
import cv2
import matplotlib.pyplot as plt

def Loc_Trung_binh_so_hoc(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros([m, n]).astype(float)
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            img_ket_qua_anh_loc[i,j] = np.mean(vung_anh_kich_thuoc_k)
    return np.uint8(img_ket_qua_anh_loc)

if __name__ == "__main__":
    img_nhieu = cv2.imread('./figures/Anh_nhieu_de_loc_Trung_binh.tif', 0)
    assert img_nhieu is not None, "khong tim thay file "
    ksize =5
    img_ket_qua_TBSH = Loc_Trung_binh_so_hoc(img_nhieu,ksize)

    fig = plt.figure(figsize=(16, 9))     # Thiết lập vùng (cửa sổ) vẽ
    (ax1, ax2) = fig.subplots(1, 2)        # Thiết lập 2 vùng con ax1, ax2
    ax1.imshow(img_nhieu, cmap='gray')      # Hiển thị ảnh gốc vùng ax1
    ax1.set_title("ảnh gốc bị nhiễu Gaussian")             # Thiết lập tiêu đề vùng ax1
    ax1.axis("off")

    ax2.imshow(img_ket_qua_TBSH, cmap='gray')       # Hiển thị ảnh sau khi lọc
    ax2.set_title("ảnh sau khi lọc Trung bình số học") # Thiết lập tiêu đề vùng ax2
    ax2.axis("off")

    plt.show()