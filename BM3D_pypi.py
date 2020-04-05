import cv2
import numpy as np
import matplotlib.pyplot as plt
import bm3d

'''
    $pip install bm3d
    https://pypi.org/project/bm3d/

    Signature:
        bm3d.bm3d(
            z: numpy.ndarray,
            sigma_psd: Union[numpy.ndarray, list, float],
            profile: Union[bm3d.profiles.BM3DProfile, str] = 'np',
            stage_arg: Union[bm3d.profiles.BM3DStages, numpy.ndarray] = <BM3DStages.ALL_STAGES: 3>,
            blockmatches: tuple = (False, False),
        ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]]
    Docstring:
        Perform BM3D denoising on z: either hard-thresholding, Wiener filtering or both.

        :param z: Noisy image. either MxN or MxNxC where C is the channel count.
                For multichannel images, blockmatching is performed on the first channel.
        :param sigma_psd: Noise PSD, either MxN or MxNxC (different PSDs for different channels)
                or
            sigma_psd: Noise standard deviation, either float, or length C list of floats
        :param profile: Settings for BM3D: BM3DProfile object or a string
                        ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb'). Default 'np'.
        :param stage_arg: Determines whether to perform hard-thresholding or wiener filtering.
                        either BM3DStages.HARD_THRESHOLDING, BM3DStages.ALL_STAGES or an estimate
                        of the noise-free image.
                        - BM3DStages.ALL_STAGES: Perform both.
                        - BM3DStages.HARD_THRESHOLDING: Perform hard-thresholding only.
                        - ndarray, size of z: Perform Wiener Filtering with stage_arg as pilot.
        :param blockmatches: Tuple (HT, Wiener), with either value either:
                            - False : Do not save blockmatches for phase
                            - True : Save blockmatches for phase
                            - Pre-computed block-matching array returned by a previous call with [True]
                            Such as y_est, matches = BM3D(z, sigma_psd, profile, blockMatches=(True, True))
                            y_est2 = BM3D(z2, sigma_psd, profile, blockMatches=matches);
        :return:
            - denoised image, same size as z: if blockmatches == (False, False)
            - denoised image, blockmatch data: if either element of blockmatches is True

'''
def PSNR(I,K):
    return 10*(np.log((255*255)/((I.astype(np.float)-K)**2).mean()))/np.log(10)


#显示图片
def plot_img(I):
   
    plt.figure(figsize = [10,8])
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    #plt.imshow(I,cmap = "gray")
    plt.show()

    #将图像元素类型转换为 uin8
def float2uint8(I):
    return np.clip(I, 0, 255).astype(np.uint)


if __name__ == "__main__":

    ori = cv2.imread("rick.jpg")  # 读入图像
    plot_img(ori)

    sigma = 30.0
    noise_rick = float2uint8(ori+np.random.randn(*ori.shape)*sigma)
    print('Noise_rick PSNR %f dB'%PSNR(ori,noise_rick ))
    plot_img(noise_rick.astype(np.uint8))
    
    #First Step: bm3d.BM3DStages.HARD_THRESHOLDING
    denoised_image = float2uint8(bm3d.bm3d(noise_rick,30,stage_arg=bm3d.BM3DStages.ALL_STAGES ))
    print ("The PSNR between the two img of the Second step is %f" % PSNR(ori, denoised_image))
    plot_img(denoised_image.astype(np.uint8))