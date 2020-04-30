'''
Retinex匀光算法

***接口函数***
Retinex(input_img,filter=['Gauss'],ksize=[3],weight=[1],gstd=0,hstd=0)
    函数说明：Retinex匀光算法(SSR/MSR)，返回处理后的图像
    参数说明：input_img 单通道图片
_BGR2HSV(input_img,model='opencv')
_HSV2BGR(h_img,s_img,v_img,model='opencv')
_HSV_channel_histogram(h_img,s_img,v_img)
    函数说明：归一化h、s、v三通道数值并显示各直方图，无返回值
_BGR2HLS(input_img,model='opencv')
_HLS2BGR(h_img,l_img,s_img,model='opencv')
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Retinex(input_img,filter=['Gauss'],ksize=[3],weight=[1],gstd=0,hstd=0): # gstd-空间高斯函数标准差;sstd-灰度值相似性高斯函数标准差
    '''Retinex匀光算法(SSR/MSR)流程:
        1.估计照度分量;2.计算反射分量;3.反射分量增强
        ***注意***
        输出input_img必须为单通道图像,即len(input_img.shape)=2
    '''
    multi_scale = len(filter)
    blur_img = []
    # 检查输入图像维度
    if(len(input_img.shape)!=2):
        raise RuntimeError('The dimensionality of input image must be 2')
    # 检查filter,ksize和weight三者维度是否一致
    if(len(filter)!=len(ksize) or len(filter)!=len(weight)):
        raise RuntimeError('The number of filter, ksize and weight are not equal')
    # 检查滤波器大小和权重和
    if(multi_scale==1):
        if (ksize[0] % 2 != 1):
            raise RuntimeError('Filter ksize must be odd')
        if(weight[0]!=1):
            raise RuntimeError('The weight must be 1')
    else:
        for i in ksize:
            if(i % 2 != 1):
                raise RuntimeError('Filter ksize must be odd')
        if(sum(weight)!=1):
            raise RuntimeError('The sum of weight must be 1')
    # 计算照度分量
    def filter_blur(input_img=input_img,filter=filter[0],ksize=ksize[0],gstd=gstd,hstd=hstd):
        if (filter == 'Gauss'):
            blur = cv2.GaussianBlur(input_img, (ksize, ksize), gstd)
        elif (filter == 'Mean'):
            blur = cv2.blur(input_img, (ksize, ksize))
        elif (filter == 'Median'):
            blur = cv2.medianBlur(input_img, ksize)
        elif (filter == 'Bilateral'):
            blur = cv2.bilateralFilter(input_img, ksize, gstd, hstd)
        else:
            raise RuntimeError('Filter type error')
        return blur
    if(multi_scale==1):
        blur_img.append(filter_blur())
    else:
        for i in range(len(filter)):
            blur_img.append(filter_blur(filter=filter[i],ksize=ksize[i]))
    # 计算反射图像:R(x,y)=f(x,y)/f(x,y)*G(x,y) => log[R(x,y)]=log[f(x,y)]-log[f(x,y)*G(x,y)] =>R(x,y)=e^{log[f(x,y)]-log[f(x,y)*G(x,y)]}
    def replace_zeroes(input_img): # 替换图像中亮度为0的像素点
        min_nonzero=min(input_img[np.nonzero(input_img)]) # np.nonzero返回表征非零元素在矩阵中位置的元组,元组中前一个列表存放非零行坐标，后一个列表存放非零元素列坐标
        input_img[input_img==0]=min_nonzero
        return input_img
    if(multi_scale==1):
        dst_input=cv2.log(replace_zeroes(input_img)/255.0) # 归一化像素值以保证cv2.log正常运行
        dst_blur=cv2.log(replace_zeroes(blur_img[0])/255.0)
        log_R=cv2.subtract(dst_input,dst_blur)
        exp_R=cv2.exp(log_R)
        dst_R=cv2.normalize(exp_R,None,0,255,cv2.NORM_MINMAX) # 归一化像素值至[0,255]
        output_img=cv2.convertScaleAbs(dst_R) # 转换为uint8格式
    else:
        intermediate_img = []
        dst_input = cv2.log(replace_zeroes(input_img) / 255.0)
        for i in range(len(blur_img)):
            dst_blur = cv2.log(replace_zeroes(blur_img[i]) / 255.0)
            log_R = cv2.subtract(dst_input, dst_blur)
            intermediate_img.append(weight[i]*log_R)
        exp_R = cv2.exp(sum(intermediate_img))
        dst_R = cv2.normalize(exp_R, None, 0, 255, cv2.NORM_MINMAX)
        output_img = cv2.convertScaleAbs(dst_R)
    # 颜色增强(暂缺)
    return output_img

def _BGR2HSV(input_img,model='opencv'):
    '''
    HSV 中的 S 控制纯色中混入白色的量，值越大，白色越少，颜色越纯；
    HSV 中的 V 控制纯色中混入黑色的量，值越大，黑色越少，明度越高；
    HLS 中的 S 和黑白没有关系，饱和度不控制颜色中混入黑白的多寡；
    HLS 中的 L 控制纯色中的混入的黑白两种颜色。'''
    # 检查输入图像通道数
    if(len(input_img.shape)!=3):
        raise RuntimeError('Input image error')
    # 采用opencv默认HSV体系:H、S、V值范围分别是[0,180]，[0,255]，[0,255]
    if(model=='opencv'):
        dst = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)  # 色彩空间转换
        h_img, s_img, v_img = cv2.split(dst)
    # 公式推算
    elif(model=='formulate'):
        pass
    else:
        raise RuntimeError('Model value error')
    return h_img,s_img,v_img # uint8格式的numpy.ndarray类型

def _HSV2BGR(h_img,s_img,v_img,model='opencv'):
    # 检查输入图像通道数
    if (len(h_img.shape)!=2 or len(s_img.shape)!=2 or len(v_img.shape)!=2):
        raise RuntimeError('Input image error')
    # 采用opencv默认HSV体系:H、S、V值范围分别是[0,180]，[0,255]，[0,255]
    if (model == 'opencv'):
        dst = cv2.merge([h_img,s_img,v_img])
        output_img = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)  # 色彩空间转换
    # 公式推算
    elif (model == 'formulate'):
        pass
    else:
        raise RuntimeError('Model value error')
    return output_img  # uint8格式的numpy.ndarray类型

def _HSV_channel_histogram(h_img,s_img,v_img):
    # 检查输入图像通道数
    if (len(h_img.shape) != 2 or len(s_img.shape) != 2 or len(v_img.shape) != 2):
        raise RuntimeError('Input image error')
    # 分通道显示
    plt.figure('hsv_h_channel')
    plt.hist(h_img*2,range=(0,360))
    plt.title('hsv_h_channel')
    plt.figure('hsv_s_channel')
    plt.hist(s_img/255.0, range=(0, 1))
    plt.title('hsv_s_channel')
    plt.figure('hsv_v_channel')
    plt.hist(v_img/255.0, range=(0, 1))
    plt.title('hsv_v_channel')
    plt.show()

def _BGR2HLS(input_img,model='opencv'):
    '''
    HSV 中的 S 控制纯色中混入白色的量，值越大，白色越少，颜色越纯；
    HSV 中的 V 控制纯色中混入黑色的量，值越大，黑色越少，明度越高；
    HLS 中的 S 和黑白没有关系，饱和度不控制颜色中混入黑白的多寡；
    HLS 中的 L 控制纯色中的混入的黑白两种颜色。'''
    # 检查输入图像通道数
    if(len(input_img.shape)!=3):
        raise RuntimeError('Input image error')
    # 采用opencv默认HLS体系
    if(model=='opencv'):
        dst = cv2.cvtColor(input_img, cv2.COLOR_BGR2HLS)  # 色彩空间转换
        h_img, l_img, s_img = cv2.split(dst)
    # 公式推算
    elif(model=='formulate'):
        pass
    else:
        raise RuntimeError('Model value error')
    return h_img,l_img,s_img # uint8格式的numpy.ndarray类型

def _HLS2BGR(h_img,l_img,s_img,model='opencv'):
    # 检查输入图像通道数
    if (len(h_img.shape)!=2 or len(l_img.shape)!=2 or len(s_img.shape)!=2):
        raise RuntimeError('Input image error')
    # 采用opencv默认HLS体系
    if (model == 'opencv'):
        dst = cv2.merge([h_img,l_img,s_img])
        output_img = cv2.cvtColor(dst, cv2.COLOR_HLS2BGR)  # 色彩空间转换
    # 公式推算
    elif (model == 'formulate'):
        pass
    else:
        raise RuntimeError('Model value error')
    return output_img  # uint8格式的numpy.ndarray类型


if __name__=='__main__':
    im = cv2.imread('paper1_1.png', 1)  # 读入<class 'numpy.ndarray'>;(height, width, channel)
    # print('channel:',len(im.shape))
    im = cv2.resize(im, (int(im.shape[1] / 5), int(im.shape[0] / 5)), cv2.INTER_AREA)  # 官方推荐缩放使用INTER_AREA(区域插值)
    # cv2.imshow('out1',im)
    channel1_img, channel2_img, channel3_img = _BGR2HSV(im)
    # print(channel1_img,channel2_img,channel3_img,sep='\n')
    # cv2.imshow('out_h_channel',channel1_img)
    # cv2.imshow('out_s_channel',channel2_img)
    # cv2.imshow('out_v_channel',channel3_img)
    # _HSV_channel_histogram(channel1_img,channel2_img,channel3_img)
    # im=Retinex(im,filter=['Gauss','Gauss'],ksize=[43,83],weight=[0.3,0.7],gstd=0,hstd=0)
    channel3_img = Retinex(channel3_img, filter=['Gauss', 'Gauss', 'Gauss'], ksize=[23, 63, 153],
                           weight=[0.1, 0.5, 0.4], gstd=0, hstd=0)
    im = _HSV2BGR(channel1_img, channel2_img, channel3_img)
    cv2.imshow('out2', im)
    # cv2.imwrite('./retinex.jpg',im)

    cv2.waitKey(0)  # 等待用户事件;参数0代表无限等待
    cv2.destroyAllWindows()  # 关闭所有窗口
