import streamlit as st
import cv2
import numpy as np


st.title("Your PhotoShop App")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

def save_image_as_png(image):
    _, buffer = cv2.imencode('.png', img=image)
    return buffer.tobytes()


if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    original_image = image
    st.image(original_image, caption="Before (Original)", use_column_width=True, channels="BGR")

    st.markdown('<p style="color:red; font-weight:bold; font-style:italic; text-decoration:underline;">Common Adjustment</p>', unsafe_allow_html=True)
    if st.checkbox('Brightness/Contrast'):
        brightness = st.slider("Brightness", -100, 100, 0, key="brightness")
        contrast = st.slider("Contrast", -100, 100, 0, key="contrast")
        image = cv2.convertScaleAbs(image, alpha=(contrast/127.0 + 1), beta=brightness)

    if st.checkbox('Gamma'):
        gamma = st.slider("gamma", 0.1, 3.0, 0.7, key="gamma")                 
        inv_gamma = 1 / gamma                                                             
        table = np.array([((i / 255) ** inv_gamma) * 255 for i in range(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)

    if st.checkbox('Satuation'):
        sat_factor = st.slider('Satuation', min_value=0.01, max_value=5.0, step=0.01, value=1.20, key='sat_factor')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] = np.clip(image[:, :, 1] * sat_factor, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


    if st.checkbox('Black-White Effect'):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    if st.checkbox('EqualizeHist'):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv = st.multiselect('YUV Selected (Y for Luma, U for Chrominance Blue, V for Chrominance Red):', options=['Y', 'U', 'V'], default=['Y'])
        if 'Y' in yuv:
            image[:,:,0] = cv2.equalizeHist(image[:,:,0])
        if 'U' in yuv:
            image[:,:,1] = cv2.equalizeHist(image[:,:,1])
        if 'V' in yuv:
            image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

    if st.checkbox('Negative Effect'):
        image = 255 - image

    if st.checkbox('Blur (Gaussian)'):
        kernel = st.slider('Kernel', min_value=1, max_value=35, step=2, value=15, key='blur_kernel')
        sigma = st.slider('Sigma', min_value=0, max_value=15, step=1, value=7, key='blur_sigma')
        image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)

    
    if st.checkbox('Sharpen'):
        ksize = st.slider('ksize', min_value=1, max_value=25, step=2, value=1, key='sharpen_ksize')
        sigma = st.slider('sigma', min_value=5, max_value=15, step=5, value=5, key='sharpen_sigma')
        b_image = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        effect = st.radio('Effect', options=['Strong', 'Normal', 'Low'], key='sharpen_effect')
        if effect=='Strong':
            image = cv2.addWeighted(src1=image, alpha=2.5, src2=b_image, beta=-1.5, gamma=0)
        elif effect=='Normal':
            image = cv2.addWeighted(src1=image, alpha=2, src2=b_image, beta=-1, gamma=0)
        elif effect=='Low':
            image = cv2.addWeighted(src1=image, alpha=1.5, src2=b_image, beta=-0.5, gamma=0)
        

    if st.checkbox('Mosaic Effect'):
        level = st.slider('Level', min_value=1, max_value=35, step=2, value=15, key='mosaic_level')
        h = int(image.shape[0]/level)                                                                                   
        w = int(image.shape[1]/level)                                                                                   
        image = cv2.resize(image, (w,h), interpolation=cv2.INTER_LINEAR)                                               
        image = cv2.resize(image, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)            


    st.image(image, caption="After", use_column_width=True, channels="BGR")
    st.image(original_image, caption="Before (Original)", use_column_width=True, channels="BGR")

    download_button = st.download_button(
        label='Download your image',
        data=save_image_as_png(image=image),
        file_name='image.png',
        key='download1'
    )
    


    ##########################################

    st.markdown('<p style="color:red; font-weight:bold; font-style:italic; text-decoration:underline;">For Edge</p>', unsafe_allow_html=True)
    if st.checkbox('Scharr'):
        dxdy = st.radio('Dx/Dy', options=['dx', 'dy'], key='scharr_dxdy')
        scale = st.slider('scale', min_value=0, max_value=10, step=1, value=1, key='scharr_scale')
        delta = st.slider('delta', min_value=0, max_value=500, step=1, value=0, key='scharr_delta')
        if dxdy == 'dx':
            image = cv2.Scharr(image, ddepth=-1, dx=1, dy=0, scale=scale, delta=delta)
        elif dxdy == 'dy':
            image = cv2.Scharr(image, ddepth=-1, dx=0, dy=1, scale=scale, delta=delta)

    if st.checkbox('Sobel'):
        dx = st.slider('dx', min_value=0, max_value=1, step=1, value=1, key='sobel_dx')
        dy = st.slider('dy', min_value=0, max_value=1, step=1, value=1, key='sobel_dy')
        ksize = st.slider('ksize', min_value=1, max_value=35, step=2, value=15, key='sobel_ksize')
        scale = st.slider('scale', min_value=1, max_value=15, step=2, value=7, key='sobel_scale')
        image = cv2.Sobel(image, ddepth=-1, dx=dx, dy=dy, ksize=ksize, scale=scale)

    if st.checkbox('Laplacian'):
        ksize = st.slider('ksize', min_value=1, max_value=15, step=2, value=1, key='lap_ksize')
        scale = st.slider('scale', min_value=0, max_value=15, step=3, value=5, key='lap_scale')
        image = cv2.Laplacian(image, ddepth=-1, ksize=ksize, scale=scale)


    if st.checkbox('Canny'):
        t1 = st.slider('Low Threshold', min_value=0, max_value=250, step=1, value=128, key='canny_t1')
        t2 = st.slider('Hihg Threshold', min_value=t1+1, max_value=255, step=1, value=255, key='canny_t2')
        image = cv2.Canny(image, threshold1=t1, threshold2=t2)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    st.image(image, caption="After", use_column_width=True, channels="BGR")
    st.image(original_image, caption="Before (Original)", use_column_width=True, channels="BGR")

    download_button = st.download_button(
        label='Download your image',
        data=save_image_as_png(image=image),
        file_name='image.png',
        key='download2'
    )


    ##############################################


    st.markdown('<p style="color:red; font-weight:bold; font-style:italic; text-decoration:underline;">For Morphology</p>', unsafe_allow_html=True)
    if st.checkbox('Erode'):
        kernel = st.slider('Kernel', min_value=1, max_value=35, step=2, value=15, key='erode_kernel')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        image = cv2.erode(image, kernel=kernel)
        

    if st.checkbox('Dilate'):
        kernel = st.slider('Kernel', min_value=1, max_value=35, step=2, value=15, key='dilate_kernel')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        image = cv2.dilate(image, kernel=kernel)

      

    if st.checkbox('Bilateral Filter'):
        diameter = st.slider('Diameter', min_value=1, max_value=25, step=2, value=9, key='bilateral_d')
        sigma_color = st.slider('Sigma for Color', min_value=1, max_value=101, step=2, value=75, key='bilateral_sc')
        sigma_space = st.slider('Sigma for Space', min_value=1, max_value=101, step=2, value=75, key='bilateral_ss')
        image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        
    if st.checkbox('Top Hat'):
        kernel = st.slider('Kernel', min_value=1, max_value=999, step=2, value=5, key='tophat_kernel')
        kernel = np.ones((kernel, kernel), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel=kernel)

    if st.checkbox('Black Hat'):
        kernel = st.slider('Kernel', min_value=1, max_value=999, step=2, value=500, key='blackhat_kernel')
        kernel = np.ones((kernel, kernel), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel=kernel)


    st.image(image, caption="After", use_column_width=True, channels="BGR")
    st.image(original_image, caption="Before (Original)", use_column_width=True, channels="BGR")

    download_button = st.download_button(
        label='Download your image',
        data=save_image_as_png(image=image),
        file_name='image.png',
        key='download3'
    )


    ##############################################


    st.markdown('<p style="color:red; font-weight:bold; font-style:italic; text-decoration:underline;">Others</p>', unsafe_allow_html=True)
    if st.checkbox('Threshold'):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thre_type = st.selectbox('Threshold Type', ('THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV'))
        thresh = st.slider('thresh', min_value=0, max_value=254, step=1, value=128, key='thresh')
        max_val = st.slider('max_val', min_value=thresh, max_value=255, step=1, value=255, key='max_value')
        if thre_type == 'THRESH_BINARY':
            ret, image = cv2.threshold(image, thresh=thresh, maxval=max_val, type=cv2.THRESH_BINARY)
        if thre_type == 'THRESH_BINARY_INV':
            ret, image = cv2.threshold(image, thresh=thresh, maxval=max_val, type=cv2.THRESH_BINARY_INV)
        if thre_type == 'THRESH_TRUNC':
            ret, image = cv2.threshold(image, thresh=thresh, maxval=max_val, type=cv2.THRESH_TRUNC)
        if thre_type == 'THRESH_TOZERO':
            ret, image = cv2.threshold(image, thresh=thresh, maxval=max_val, type=cv2.THRESH_TOZERO)
        if thre_type == 'THRESH_TOZERO_INV':
            ret, image = cv2.threshold(image, thresh=thresh, maxval=max_val, type=cv2.THRESH_TOZERO_INV)
        
    if st.checkbox('Adaptive Threshold'):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adap_thre_type = st.selectbox('Adpat. Threshold Type', ('ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C'))
        max_val = st.slider('max_val', min_value=128, max_value=255, step=1, value=255, key='max_value')
        blocksize = st.slider('blocksize', min_value=1, max_value=19, step=2, value=11, key='blocksize')
        C = st.slider('C', min_value=1, max_value=10, step=1, value=2, key='C')
        if adap_thre_type == 'ADAPTIVE_THRESH_MEAN_C':
            image = cv2.adaptiveThreshold(image, maxValue=max_val, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=blocksize, C=C)
        if adap_thre_type == 'ADAPTIVE_THRESH_GAUSSIAN_C':
            image = cv2.adaptiveThreshold(image, maxValue=max_val, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=blocksize, C=C)
 
    st.image(image, caption="After", use_column_width=True)
    st.image(original_image, caption="Before (Original)", use_column_width=True, channels="BGR")

    download_button = st.download_button(
        label='Download your image',
        data=save_image_as_png(image=image),
        file_name='image.png',
        key='download4'
    )
