import time
from operator import itemgetter

import cv2
import cv2 as cv
import lasio
import matplotlib.pyplot as plt
import numpy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize
import streamlit as st
#from streamlit_plotly_events import plotly_events
from utils import resort




st.title('Semiautomatic LWD image log interpretation using deep learning')


@st.cache_data
def read_las(path, name, plot=False):
    data = lasio.read(path + name).data
    data = data[:, :-3]
    a = np.random.randint(0, data.shape[0] // 2)
    if plot:
        plt.figure(figsize=[10, 30])
        plt.imshow(data[a:a + 20, 1:], aspect='auto', cmap='afmhot_r')
        plt.show()
    return data, a


def smoothing(data, a, plot=False):
    kernel = np.ones((5, 5), np.float32) / 25
    data_smooth = cv.filter2D(data[:, 1:], -1, kernel)
    if plot:
        # plt.figure(figsize=[10, 30])
        fig = plt.imshow(data_smooth[a:a + 20, 1:], aspect='auto', cmap='afmhot_r')
        plt.title('Sample:' + str(a))
        plt.show()
    return data_smooth


def reshape(data_smooth, target_bin, plot=False):
    binnable = (data_smooth.shape[0] // 20) * 20
    data_smooth = data_smooth[:binnable, :]
    data_resh = np.reshape(data_smooth, (data_smooth.shape[0] // target_bin, target_bin, data_smooth.shape[1]))
    if plot:
        plt.figure(figsize=[10, 30])
        plt.imshow(data_resh[625, 1:], aspect='auto', cmap='afmhot_r')
        plt.title('Bin 625')
        # plt.show()

    return data_resh


data_load_state = st.text('loading data...')
realdata_dir = "data/CORAL_8/LAS/"
name = 'CORAL_08_DIR_DYNAMIC.las'
data, a = read_las(realdata_dir, name)
data = smoothing(data, a, plot=False)
data_r = reshape(data, 20)
data_load_state.text('data successfully loaded.')


def welcome():


    st.subheader('A project by the Exploration Geophysics Group (EGG) ')
    st.subheader('University of Trieste ')
    # st.write()

    # st.subheader('1 meter of LWD density data')
    wlcm_img = cv2.imread('resources/logo egg (2).jpg')
    wlcm_img = cv2.resize(wlcm_img, (300, 350))
    fig = px.imshow(wlcm_img, color_continuous_scale='binary')
    # st.plotly_chart(fig, aspect = 'equal')
    st.image(wlcm_img, use_column_width=False)


def photo():
    st.subheader('Raw LWD data')
    st.write(data[a:a + 20, :])

    st.subheader('Manual processing workflow')
    "Click the pixels in the displayed image. For each selected pixel click on the 'Pick the selected point' button to confirm the selection."
    "The borehole image logs are unwrapped visualizations of the 360° of the well. These boreholes are ideally crossed by geological planes that we find in form of sinusoids in the images."
    "The manual processing of these data involve a picking phase, i.e., the selection of points located along a geological contrast. So, we should select the pixels in the image that lie (in our opinion) on a formation change."

    "Seek the sinusoidal pattern those pixels should follow, if any"
    # fig = px.imshow(data[a:a + 20, :], aspect='auto', color_continuous_scale='hot')
    # fig_pick = plotly_events(fig, click_event= True) #, , select_event= True, hover_event=True
    # print('fig_pick', fig_pick)
    # b = fig_pick[0]
    # print('b',b)
    # b = pd.DataFrame.from_dict(b, orient='index')
    # b['pointIndex']

    st.subheader('1 meter of LWD density data')
    if 'fig' not in st.session_state:
        st.session_state.figure = px.imshow(data[a:a + 20, :], aspect='auto', color_continuous_scale='hot')
        st.session_state.figure_pick = plotly_events(st.session_state.figure, click_event=True)
        try:
            st.session_state.b = st.session_state.figure_pick[0]
        except:
            print("Waiting for pick...")
        pick = st.button('Pick selected point')

    if 'pointsx' not in st.session_state:
        st.session_state.pointsx = []
    if 'pointsy' not in st.session_state:
        st.session_state.pointsy = []

    if pick:
        st.session_state.pointsx.append(st.session_state.b['pointIndex'][1])
        st.session_state.pointsy.append(st.session_state.b['pointIndex'][0])
        # using sorted() + zip() + itemgetter()
        # integrity sorting in two list
        res = [list(x) for x in zip(*sorted(zip(st.session_state.pointsx, st.session_state.pointsy),
                                            key=itemgetter(0)))]
        st.session_state.pointsx, st.session_state.pointsy = res[0], res[1]
        st.session_state.figure = st.session_state.figure
        st.session_state.figure.add_trace(
            go.Scatter(x=st.session_state.pointsx, y=st.session_state.pointsy, mode='markers',
                       marker=dict(color=['black'])))
        # st.session_state.figure.show()
        # fig.add_trace(go.Scatter(x=st.session_state.pointsx,y=st.session_state.pointsy, mode='markers', marker=dict(color=['black'])))
        st.session_state.figure_pick = plotly_events(st.session_state.figure, click_event=True)

    c1, c2 = st.columns(2)
    with c1:
        '''list of x coordinates'''
        st.session_state.pointsx
    with c2:
        '''list of y coordinates'''
        st.session_state.pointsy

    "After the picking, normally the processing softwares are equipped with functions that perform a non-linear regression of the picked points to get the best fitted sinusoid. " \
    "Click on the 'Correlate feature' button to see a similar result on your picking outcome!"

    corr = st.button("Correlate feature")
    reset = st.button("Reset")

    if corr:
        try:
            st.session_state.sin = np.zeros((16))
            x_data = np.linspace(0, 2 * np.pi, 16)

            def fit_sin(tt, yy):
                # '''Fit sin to the input sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
                tt = numpy.array(tt)
                yy = numpy.array(yy)
                ff = numpy.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
                Fyy = abs(numpy.fft.fft(yy))
                guess_freq = abs(
                    ff[numpy.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
                guess_amp = numpy.std(yy) * 2. ** 0.5
                guess_offset = numpy.mean(yy)
                guess = numpy.array([guess_amp, 2. * numpy.pi * guess_freq, 0., guess_offset])

                def sinfunc(t, A, w, p, c):  return A * numpy.sin(w * t + p) + c

                popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=5000)
                A, w, p, c = popt
                f = w / (2. * numpy.pi)
                fitfunc = lambda t: A * numpy.sin(w * t + p) + c
                return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
                        "maxcov": numpy.max(pcov), "rawres": (guess, popt, pcov)}

            sin = fit_sin(x_data, st.session_state.pointsy)
            st.session_state.sin = sin['fitfunc'](x_data)
            st.session_state.sin

            st.session_state.figure.add_trace(go.Scatter(x=np.linspace(0, 15, 16), y=st.session_state.sin, mode='lines'))
            st.session_state.figure_pick = plotly_events(st.session_state.figure, click_event=True)
        except:
            st.markdown("##### ERROR: You must select more points to use the function")
    if reset:
        st.session_state.pointsx = []
        st.session_state.pointsy = []

    "As we can see, the rosultion of this type of LWD image logs is quite low."
    "We can also note that when it comes to pick the points in the image, we are not that sure about where to select the real contrast."
    "Finally, imagine how time-consuming can be to do this operation for example for a 600 meters interval. Imagine the struggle of a petrophysicist or a mud logger trying to accurately process and interpret these data in real-time konwing that the Rate of Penetration (ROP) normally is around 25 ft/hr.  "


def video():
    st.subheader('PickNet: a U-Net architecture for LWD image logs segmentation')
    "We use a U-Net architecture to automate the picking step. U-net is a state-of the art deep learning architecture very effective in segmentation of images, i.e., to classifiy the pixels in the image according the categories of object that we have to detect in the images."
    "This network was trained ono synthetic data to classify pixel in the LWD images as 0s or 1s, where they're associate to a 0 it means there's no evident formation change in the image, otherwise it means that the pixels are located on a geological edge. "
    c1, c2 = st.columns(2)
    with c1:
        ''''''
        video_file = open('resources/SL animation.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

    with c2:
        ''''''
        video_file = open('resources/CL animation (1).mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

    st.subheader('FitNet-05: a FCNN architecture for feature correlation')

    "We trained FitNet to correlate the feature performing non-linear regression of the segmented edges, to retrieve the associated sinusoids. We see the results on the PickNet segmentation results trained with different strategies. "

    c1, c2 = st.columns(2)
    with c1:
        ''''''
        video_file = open('resources/8_P05_T150 sl.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

    with c2:
        ''''''
        video_file = open('resources/8_p05_T150 (4).mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)



def feature_detection():

    CL = np.load("C:/Desktop/DOTTORATO/WORK/001_Deep_Learning/01_Curve-Net/output/restored segmentation maps/restoreCL.npy")

    SL = np.load("C:/Desktop/DOTTORATO/WORK/001_Deep_Learning/01_Curve-Net/output/restored segmentation maps/restoreSL.npy")

    fitnet_CL = np.load("C:/Desktop/DOTTORATO/WORK/001_Deep_Learning/01_Curve-Net/output/FitNet05 on segmentation maps/Predictions_CL8.npy")

    fitnet_SL =  np.load("C:/Desktop/DOTTORATO/WORK/001_Deep_Learning/01_Curve-Net/output/FitNet05 on segmentation maps/Predictions_SL8.npy")

    if 'preds_segmCL' not in st.session_state:

        st.session_state.preds_segmCL= CL

    if 'preds_segmSL' not in st.session_state:

        st.session_state.preds_segmSL = SL

    if 'preds_fitCL' not in st.session_state:

        st.session_state.preds_fitCL= fitnet_CL

    if 'preds_fitSL' not in st.session_state:

        st.session_state.preds_fitSL = fitnet_SL

    def resort(i):

        preds = []
        matrix = []
        preds1 = []

        preds2 = []
        preds21 = []
        matrix.append(data[i:i+40,:])
        matrix = np.asarray(matrix)
        print(matrix[0].shape)

        p0 = np.array([p[i] for p in fitnet_CL])
        pst = p0.mean(axis=0)[0]
        pst1 = p0.mean(axis=0)[1]

        p1 = np.array([p[i] for p in fitnet_SL])
        pst2 = p1.mean(axis=0)[0]
        pst21 = p1.mean(axis=0)[1]

        m, s = p0.mean(axis=0)[0], p0.std(axis=0)[0]
        m1,s1 = p0.mean(axis=0)[1], p0.std(axis=0)[1]
        m2, s2 = p1.mean(axis=0)[0], p1.std(axis=0)[0]
        m21, s21 = p1.mean(axis=0)[1], p1.std(axis=0)[1]

        preds.append(pst)
        preds1.append(pst1)
        preds = np.asarray(preds)

        preds2.append(pst2)
        preds21.append(pst21)
        preds2 = np.asarray(preds2)

        return preds, preds1, preds2, preds21, matrix, [(m,s), (m1,s1), (m2,s2), (m21, s21)]


    start = st.button('Start')


    if start:

        import time

        # Funzione per aggiornare l'animazione
        def update_animation(fig, matrix, x_curve, y_curve):

            i=j
            preds = resort(i)[0][0]
            preds1 = resort(i)[1][0]

            preds2 = resort(i)[2][0]
            preds21 = resort(i)[3][0]

            print(preds)
            matrix = resort(i)[4][0]

            pars = resort(i)[5]

            #st.session_state.figure = px.imshow(matrix, aspect='auto', color_continuous_scale='hot')
            st.session_state.figure.data[0]['z'] = matrix


            st.session_state.figure.data[1]['x'],st.session_state.figure.data[1]['y'] = np.linspace(0,15,16), preds
            st.session_state.figure.data[2]['x'],st.session_state.figure.data[2]['y'] =np.linspace(0,15,16), preds1
            st.session_state.figure.data[3]['x'],st.session_state.figure.data[3]['y'] =np.linspace(0,15,16), preds2
            st.session_state.figure.data[4]['x'],st.session_state.figure.data[4]['y'] = np.linspace(0,15,16), preds21
            st.session_state.figure_pick = plotly_events(st.session_state.figure, click_event=True)
            print(st.session_state.figure)
        # Creazione dell'interfaccia Streamlit
        st.title('LWD image with overlayed DL correlated feature')
        #rows = st.slider('Number of Rows:', 10, 100, 50)
        #cols = st.slider('Number of Columns:', 10, 100, 50)
        #num_points = 100

       # curve = next(preds_cycle)
        preds = resort(0)[0][0]
        matrix = resort(0)[4][0]
        x_curve = np.linspace(0,15, 16)

        st.session_state.figure = px.imshow(matrix, aspect='auto', color_continuous_scale='hot')
        st.session_state.figure.add_trace(
            go.Scatter(x=np.linspace(0, 15, 16), y=np.zeros((16)), mode='lines+markers', line_color='#ffe476'))
        st.session_state.figure.add_trace(
            go.Scatter(x=np.linspace(0, 15, 16), y=np.zeros((16)), mode='lines+markers', line_color='#ffe476'))
        st.session_state.figure.add_trace(
            go.Scatter(x=np.linspace(0, 15, 16), y=np.zeros((16)), mode='lines+markers', line=dict(color="#0000ff")))
        st.session_state.figure.add_trace(
            go.Scatter(x=np.linspace(0, 15, 16), y=np.zeros((16)), mode='lines+markers', line=dict(color="#0000ff")))



        #st.session_state.figure_pick = plotly_events(st.session_state.figure, click_event=True)

        for j in range(9800,9900):  # Numero di frame dell'animazione

            update_animation(st.session_state.figure, matrix, x_curve, preds)
            #st.pyplot(fig, use_container_width=True)
            #time.sleep(animation_interval)
            st.session_state.figure.update()


    if 'confirmed' not in st.session_state:
        st.session_state.confirmed = []



def main():
    selected_box = st.sidebar.selectbox(
        'Choose one of the following',
        ('Welcome', 'Image Processing', 'Video',  'Feature Detection', 'plotly anim')
    )


    if selected_box == 'Welcome':
        welcome()
    if selected_box == 'Image Processing':
        photo()
    if selected_box == 'Video':
        video()
    if selected_box == 'Feature Detection':
        feature_detection()



if __name__ == "__main__":
    main()


