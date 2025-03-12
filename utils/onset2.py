import keyword
import os
import re
from turtle import st
from biosppy import storage
import numpy as np
import collections
import sys
import torch

from utils.extract_feature import get_feature_window

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将根目录添加到 sys.path
sys.path.append(root_dir)



#from HandInteract.utils.dataset_construction import pdf_construct_dataset, format_folders
#from HandInteract.utils.visualization import plot_and_save_csv
from filter3 import *
from matplotlib import pyplot as plt
from scipy.signal import windows as ss_win  # 新增导入
from scipy import signal as ss
import six
from scipy.stats import stats, kurtosis,skew


output_dir = r"D:\download\feishu_download\fragment4"
outputplt_dir = r"D:\download\feishu_download\fragment4可视化文件"
data_path = r"D:\download\feishu_download\李佳乐2"
dataplt_dir = r"D:\download\feishu_download\李佳乐2可视化文件"

MAJOR_LW = 2.5
MINOR_LW = 1.5
MAX_ROWS = 10


class ReturnTuple(tuple):
    """A named tuple to use as a hybrid tuple-dict return object.

    Parameters
    ----------
    values : iterable
        Return values.
    names : iterable, optional
        Names for return values.

    Raises
    ------
    ValueError
        If the number of values differs from the number of names.
    ValueError
        If any of the items in names:
        * contain non-alphanumeric characters;
        * are Python keywords;
        * start with a number;
        * are duplicates.

    """

    def __new__(cls, values, names=None):

        return tuple.__new__(cls, tuple(values))

    def __init__(self, values, names=None):

        nargs = len(values)

        if names is None:
            # create names
            names = ["_%d" % i for i in range(nargs)]
        else:
            # check length
            if len(names) != nargs:
                raise ValueError("Number of names and values mismatch.")

            # convert to str
            names = list(map(str, names))

            # check for keywords, alphanumeric, digits, repeats
            seen = set()
            for name in names:
                if not all(c.isalnum() or (c == "_") for c in name):
                    raise ValueError(
                        "Names can only contain alphanumeric \
                                      characters and underscores: %r."
                        % name
                    )

                if keyword.iskeyword(name):
                    raise ValueError("Names cannot be a keyword: %r." % name)

                if name[0].isdigit():
                    raise ValueError("Names cannot start with a number: %r." % name)

                if name in seen:
                    raise ValueError("Encountered duplicate name: %r." % name)

                seen.add(name)

        self._names = names

    def as_dict(self):
        """Convert to an ordered dictionary.

        Returns
        -------
        out : OrderedDict
            An OrderedDict representing the return values.

        """

        return collections.OrderedDict(zip(self._names, self))

    __dict__ = property(as_dict)

    def __getitem__(self, key):
        """Get item as an index or keyword.

        Returns
        -------
        out : object
            The object corresponding to the key, if it exists.

        Raises
        ------
        KeyError
            If the key is a string and it does not exist in the mapping.
        IndexError
            If the key is an int and it is out of range.

        """

        if isinstance(key, six.string_types):
            if key not in self._names:
                raise KeyError("Unknown key: %r." % key)

            key = self._names.index(key)

        return super(ReturnTuple, self).__getitem__(key)

    def __repr__(self):
        """Return representation string."""

        tpl = "%s=%r"

        rp = ", ".join(tpl % item for item in zip(self._names, self))

        return "ReturnTuple(%s)" % rp

    def __getnewargs__(self):
        """Return self as a plain tuple; used for copy and pickle."""

        return tuple(self)

    def keys(self):
        """Return the value names.

        Returns
        -------
        out : list
            The keys in the mapping.

        """

        return list(self._names)


def normpath(path):
    """Normalize a path.

    Parameters
    ----------
    path : str
        The path to normalize.

    Returns
    -------
    npath : str
        The normalized path.

    """

    if "~" in path:
        out = os.path.abspath(os.path.expanduser(path))
    else:
        out = os.path.abspath(path)

    return out



def plot_emg(ts_filter=None,
             ts_raw=None,
             sampling_rate=None,
             raw=None,
             filtered=None,
             onsets=None,
             processed=None,
             path=None,
             show=False,
             smooth=None):
    """Create a summary plot from the output of signals.emg.emg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    sampling_rate : int, float
        Sampling frequency (Hz).
    raw : array
        Raw EMG signal.
    filtered : array
        Filtered EMG signal.
    onsets : array
        Indices of EMG pulse onsets.
    processed : array, optional
        Processed EMG signal according to the chosen onset detector.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('EMG Summary')

    if processed is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313)


        # processed signal
        L = len(processed)
        T = (L - 1) / sampling_rate
        ts_processed = np.linspace(0, T, L, endpoint=True)
        ax3.plot(ts_processed, processed,
                 linewidth=MAJOR_LW,
                 label='Processed')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.legend()
        ax3.grid()
    else:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax3.plot(ts_filter, smooth, linewidth=MAJOR_LW, label='Smooth')

        ax3.set_ylabel('Amplitude')
        ax3.legend()
        ax3.grid()

    # raw signal
    ax1.plot(ts_raw, raw, linewidth=MAJOR_LW, label='Raw')

    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid()



    # filtered signal with onsets
    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts_filter, filtered, linewidth=MAJOR_LW, label='Filtered')
    ax2.vlines(ts_filter[onsets], ymin, ymax,
               color='m',
               linewidth=MINOR_LW,
               label='Onsets')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid()

    # make layout tight
    fig.tight_layout()

    # save to file
    if path is not None:
        path = normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)

def plot_emg2(ts_fragment=None,

             sampling_rate=None,

             fragment=None,
             onsets=None,
             processed=None,
             path=None,
             show=False,
             smooth=None):
    """Create a summary plot from the output of signals.emg.emg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    sampling_rate : int, float
        Sampling frequency (Hz).
    raw : array
        Raw EMG signal.
    filtered : array
        Filtered EMG signal.
    onsets : array
        Indices of EMG pulse onsets.
    processed : array, optional
        Processed EMG signal according to the chosen onset detector.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('EMG Summary')

    if processed is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313)


        # processed signal
        L = len(processed)
        T = (L - 1) / sampling_rate
        ts_processed = np.linspace(0, T, L, endpoint=True)
        ax3.plot(ts_processed, processed,
                 linewidth=MAJOR_LW,
                 label='Processed')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.legend()
        ax3.grid()
    else:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(ts_fragment, fragment, linewidth=MAJOR_LW, label='Fragment')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid()
        ax2.plot(ts_fragment, smooth, linewidth=MAJOR_LW, label='Smooth')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid()




    # make layout tight
    fig.tight_layout()

    # save to file
    if path is not None:
        path = normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=500, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)


def data_processing(dataset_path, is_cwt,sampling_rate,show):
    """
    规整化数据，滤波后转为列表
    :param dataset_path: 数据集路径
    :param is_cwt:是否进行小波变换分析。如果要进行，则不进行之后的所有操作，因为小波变换是为了确认滤波的频率
    :return:[归一化均值为0，方差为1的信号，torch.tensor], [类别，string]一一对应的两个一维列表
    """
    categories = os.listdir(dataset_path)
    signals_raw = []
    signals_rest = []
    signals_filter = []
    labels = []
    output = []

    i=0

    for idx, category in enumerate(categories):
        path = os.path.join(dataset_path, category)
        output.append([])
        labels.append([])

        for file_name in os.listdir(path):
            file = os.path.join(dataset_path, category, file_name)
            df = pd.read_csv(file)  # 读取信号
            origin_signal = df.iloc[50000:, 1].values.astype(np.float32)

            origin_signal = np.reshape(origin_signal,(-1,1))#(1792689,1)
            data_raw = torch.tensor(origin_signal, dtype=torch.float32)
            filtered_signal = signal_filter(origin_signal)  # 滤波(7171,1)
            # 归一化
            data_filter = torch.tensor(filtered_signal, dtype=torch.float32)
            normalized_data_filter = (data_filter - torch.mean(data_filter)) / torch.std(data_filter)
            normalized_data_raw = (data_raw - torch.mean(data_raw)) / torch.std(data_raw)
            if i==0:
                normalized_data_rest = normalized_data_filter[100:200]#(100,1)取第1-2秒的信号作为rest
                i=1
            # 添加到列表内
            signals_filter.append(normalized_data_filter)
            signals_raw.append(normalized_data_raw)
            signals_rest.append(normalized_data_rest)
            #labels.append(category)


            x_filter = normalized_data_filter.detach().cpu().numpy()
            x_raw = normalized_data_raw.detach().cpu().numpy()
            x_rest = normalized_data_rest.detach().cpu().numpy()
            rest_signal = x_rest[:] - np.mean(x_rest[:])
            var_rest = np.var(rest_signal)
            threshold1 = var_rest * 0.5 + np.mean(x_rest[:])
            threshold2 = var_rest * 0.25 + np.mean(x_rest[:])

            # find onsets
            #smooth,onsets = find_onsets(signal=x_filter, sampling_rate=sampling_rate, size=0.05, threshold=threshold)

            onsets2, tflist, smooth = bonato_onset_detector(signal=x_filter, rest=x_rest, sampling_rate=sampling_rate,
                                                            threshold1=threshold1,threshold2=threshold2, active_state_duration_begin=3,active_state_duration_end = 33,
                                                            samples_above_fail=8, fail_size=10)


            # get time vectors
            length_filter = len(x_filter)
            T1 = (length_filter - 1) / sampling_rate
            ts_filter = np.linspace(0, T1, length_filter, endpoint=True)
            length_raw = len(x_raw)
            T2 = (length_raw - 1) / sampling_rate
            ts_raw = np.linspace(0, T2, length_raw, endpoint=True)

            #fragment = x_filter[onsets[0]:onsets[len(onsets)-1]]
            for k in range(0, len(onsets2), 2):
                if k + 1 == len(onsets2):
                    break
                fragment = x_filter[onsets2[k]:onsets2[k + 1]]
                m = re.match(r"^(\d)\.csv$", file_name).group(1)
                # 生成输出路径
                os.makedirs(os.path.join(output_dir, category), exist_ok=True)

                output_path = os.path.join(output_dir, category, f'{m}.{k / 2}.csv')
                # 保存片段
                os.makedirs(os.path.join(outputplt_dir, category), exist_ok=True)

                outputplt_path = os.path.join(outputplt_dir, category, f'{m}.{k / 2}.csv')
                pd.DataFrame(fragment).to_csv(output_path, index=False, header=False)
                print(f"已保存文件：{m}.{k / 2}.csv")
                length_fragment = len(fragment)
                T3 = (length_fragment - 1) / sampling_rate
                ts_fragment = np.linspace(0, T3, length_fragment, endpoint=True)
                onset = (onsets2[k], onsets2[k + 1])

                smooth_fragment = smooth[onsets2[k]:onsets2[k + 1]]
                plot_emg2(ts_fragment=ts_fragment,

                          sampling_rate=10000.,

                          fragment=fragment,
                          processed=None,
                          onsets=onset,
                          path=outputplt_path,
                          show=show,
                          smooth=smooth_fragment)

            

            os.makedirs(os.path.join(dataplt_dir, category), exist_ok=True)

            outputplt_path = os.path.join(dataplt_dir, category, file_name)
            # plot
            '''
            onsets_all = []
            onsets_all.append(onsets[0])
            onsets_all.append(onsets[len(onsets)-1])
            '''
            plot_emg(ts_filter=ts_filter,
                     ts_raw=ts_raw,
                     sampling_rate=100.,
                     raw=x_raw,
                     filtered=x_filter,
                     processed=None,
                     onsets=onsets2,
                     path=outputplt_path,
                     show=show,
                     smooth=smooth)

            # 滑动窗口+特征提取####还没改，后面考虑每个动作再加滑窗
            #feature_windows, window_num = get_feature_window(fragment, window_size=32)
            # 添加到列表内
            #output[-1].extend(feature_windows)
            #labels[-1].extend([idx] * window_num)
    return output, labels



def _get_window(kernel, size, **kwargs):
    """Return a window with the specified parameters.

    Parameters
    ----------
    kernel : str
        Type of window to create.
    size : int
        Size of the window.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.windows function.

    Returns
    -------
    window : array
        Created window.

    """

    # mimics scipy.signal.get_window
    if kernel in ["blackman", "black", "blk"]:
        winfunc = ss.blackman
    elif kernel in ["triangle", "triang", "tri"]:
        winfunc = ss.triang
    elif kernel in ["hamming", "hamm", "ham"]:
        winfunc = ss_win.hamming
    elif kernel in ["bartlett", "bart", "brt"]:
        winfunc = ss.bartlett
    elif kernel in ["hanning", "hann", "han"]:
        winfunc = ss_win.hann
    elif kernel in ["blackmanharris", "blackharr", "bkh"]:
        winfunc = ss.blackmanharris
    elif kernel in ["parzen", "parz", "par"]:
        winfunc = ss_win.parzen
    elif kernel in ["bohman", "bman", "bmn"]:
        winfunc = ss.bohman
    elif kernel in ["nuttall", "nutl", "nut"]:
        winfunc = ss.nuttall
    elif kernel in ["barthann", "brthan", "bth"]:
        winfunc = ss.barthann
    elif kernel in ["flattop", "flat", "flt"]:
        winfunc = ss.flattop
    elif kernel in ["kaiser", "ksr"]:
        winfunc = ss.kaiser
    elif kernel in ["gaussian", "gauss", "gss"]:
        winfunc = ss.gaussian
    elif kernel in [
        "general gaussian",
        "general_gaussian",
        "general gauss",
        "general_gauss",
        "ggs",
    ]:
        winfunc = ss.general_gaussian
    elif kernel in ["boxcar", "box", "ones", "rect", "rectangular"]:
        winfunc = ss_win.boxcar
    elif kernel in ["slepian", "slep", "optimal", "dpss", "dss"]:
        winfunc = ss.slepian
    elif kernel in ["cosine", "halfcosine"]:
        winfunc = ss.cosine
    elif kernel in ["chebwin", "cheb"]:
        winfunc = ss.chebwin
    else:
        raise ValueError("Unknown window type.")

    try:
        window = winfunc(size, **kwargs)
    except TypeError as e:
        raise TypeError("Invalid window arguments: %s." % e)

    return window


def smoother(signal=None, kernel="boxzen", size=10, mirror=True, **kwargs):
    """Smooth a signal using an N-point moving average [MAvg]_ filter.

    This implementation uses the convolution of a filter kernel with the input
    signal to compute the smoothed signal [Smit97]_.

    Availabel kernels: median, boxzen, boxcar, triang, blackman, hamming, hann,
    bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
    kaiser (needs beta), gaussian (needs std), general_gaussian (needs power,
    width), slepian (needs width), chebwin (needs attenuation).

    Parameters
    ----------
    signal : array
        Signal to smooth.
    kernel : str, array, optional
        Type of kernel to use; if array, use directly as the kernel.
    size : int, optional
        Size of the kernel; ignored if kernel is an array.
    mirror : bool, optional
        If True, signal edges are extended to avoid boundary effects.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.windows function.

    Returns
    -------
    signal : array
        Smoothed signal.
    params : dict
        Smoother parameters.

    Notes
    -----
    * When the kernel is 'median', mirror is ignored.

    References
    ----------
    .. [MAvg] Wikipedia, "Moving Average",
       http://en.wikipedia.org/wiki/Moving_average
    .. [Smit97] S. W. Smith, "Moving Average Filters - Implementation by
       Convolution", http://www.dspguide.com/ch15/1.htm, 1997

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to smooth.")

    length = len(signal)

    if isinstance(kernel, six.string_types):
        # check length
        if size > length:
            size = length - 1

        if size < 1:
            size = 1

        if kernel == "boxzen":
            # hybrid method
            # 1st pass - boxcar kernel
            aux, _ = smoother(signal, kernel="boxcar", size=size, mirror=mirror)

            # 2nd pass - parzen kernel
            smoothed, _ = smoother(aux, kernel="parzen", size=size, mirror=mirror)

            params = {"kernel": kernel, "size": size, "mirror": mirror}

            args = (smoothed, params)
            names = ("signal", "params")

            return ReturnTuple(args, names)

        elif kernel == "median":
            # median filter
            if size % 2 == 0:
                raise ValueError("When the kernel is 'median', size must be odd.")

            smoothed = ss.medfilt(signal, kernel_size=size)

            params = {"kernel": kernel, "size": size, "mirror": mirror}

            args = (smoothed, params)
            names = ("signal", "params")

            return ReturnTuple(args, names)

        else:
            win = _get_window(kernel, size, **kwargs)

    elif isinstance(kernel, np.ndarray):
        win = kernel
        size = len(win)

        # check length
        if size > length:
            raise ValueError("Kernel size is bigger than signal length.")

        if size < 1:
            raise ValueError("Kernel size is smaller than 1.")

    else:
        raise TypeError("Unknown kernel type.")

    # convolve
    w = win / win.sum()
    if mirror:
        aux = np.concatenate(
            (signal[0] * np.ones(size), signal, signal[-1] * np.ones(size))
        )
        smoothed = np.convolve(w, aux, mode="same")
        smoothed = smoothed[size:-size]
    else:
        smoothed = np.convolve(w, signal, mode="same")

    # output
    params = {"kernel": kernel, "size": size, "mirror": mirror}
    params.update(kwargs)

    args = (smoothed, params)
    names = ("signal", "params")

    return ReturnTuple(args, names)


def find_onsets(signal=None, sampling_rate=1000., size=0.05, threshold=None):
    """Determine onsets of EMG pulses.

    Skips corrupted signal parts.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Detection window size (seconds).
    threshold : float, optional
        Detection threshold.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # full-wave rectification
    fwlo = np.abs(signal)

    # smooth
    size = int(sampling_rate * size)
    mvgav, _ = smoother(signal=fwlo,
                           kernel='boxzen',
                           size=size,
                           mirror=True)
    mvgav2, _ = smoother(signal=signal,
                        kernel='boxzen',
                        size=size,
                        mirror=True)

    # threshold
    if threshold is None:
        aux = np.abs(mvgav)
        threshold = 1.2 * np.mean(aux) + 2.0 * np.std(aux, ddof=1)

    # find onsets
    length = len(signal)
    start = np.nonzero(mvgav > threshold)[0]
    stop = np.nonzero(mvgav <= threshold)[0]

    onsets = np.union1d(np.intersect1d(start - 1, stop),
                        np.intersect1d(start + 1, stop))

    if np.any(onsets):
        if onsets[-1] >= length:
            onsets[-1] = length - 1

    return mvgav2,onsets

def signal_stats(signal=None):
    """Compute various metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    mean : float
        Mean of the signal.
    median : float
        Median of the signal.
    min : float
        Minimum signal value.
    max : float
        Maximum signal value.
    max_amp : float
        Maximum absolute signal amplitude, in relation to the mean.
    var : float
        Signal variance (unbiased).
    std_dev : float
        Standard signal deviation (unbiased).
    abs_dev : float
        Mean absolute signal deviation around the median.
    kurtosis : float
        Signal kurtosis (unbiased).
    skew : float
        Signal skewness (unbiased).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # mean
    mean = np.mean(signal)

    # median
    median = np.median(signal)

    # min
    minVal = np.min(signal)

    # max
    maxVal = np.max(signal)

    # maximum amplitude
    maxAmp = np.abs(signal - mean).max()

    # variance
    sigma2 = signal.var(ddof=1)

    # standard deviation
    sigma = signal.std(ddof=1)

    # absolute deviation
    ad = np.mean(np.abs(signal - median))

    # kurtosis
    kurt = kurtosis(signal, bias=False)

    # skweness
    skweness = skew(signal, bias=False)

    # output
    args = (mean, median, minVal, maxVal, maxAmp, sigma2, sigma, ad, kurt, skweness)
    names = (
        "mean",
        "median",
        "min",
        "max",
        "max_amp",
        "var",
        "std_dev",
        "abs_dev",
        "kurtosis",
        "skewness",
    )

    return ReturnTuple(args, names)


def bonato_onset_detector(signal=None, rest=None, sampling_rate=1000.,
                          threshold1=None,threshold2=None, active_state_duration_begin=None,active_state_duration_end=None,
                          samples_above_fail=None, fail_size=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Bonato et al. [Bo98]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    rest : array, list, dict
        One of the following 3 options:
        * N-dimensional array with filtered samples corresponding to a
        rest period;
        * 2D array or list with the beginning and end indices of a segment of
        the signal corresponding to a rest period;
        * Dictionary with {'mean': mean value, 'std_dev': standard variation}.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : int, float
        Detection threshold.
    active_state_duration: int
        Minimum duration of the active state.
    samples_above_fail : int
        Number of samples above the threshold level in a group of successive
        samples.
    fail_size : int
        Number of successive samples.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [Bo98] Bonato P, D’Alessio T, Knaflitz M, "A statistical method for the
       measurement of muscle activation intervals from surface myoelectric
       signal during gait", IEEE Transactions on Biomedical Engineering,
       vol. 45:3, pp. 287–299, 1998

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rest is None:
        raise TypeError("Please specidy rest parameters.")




    if samples_above_fail is None:
        raise TypeError("Please specify the number of samples above the "
                        "threshold level in a group of successive samples.")

    if fail_size is None:
        raise TypeError("Please specify the number of successive samples.")

    # gather statistics on rest signal
    if isinstance(rest, np.ndarray) or isinstance(rest, list):
        # if the input parameter is a numpy array or a list
        if len(rest) >= 2:
            # first ensure numpy
            rest = np.array(rest)
            if len(rest) == 2:
                # the rest signal is a segment of the signal
                rest_signal = signal[rest[0]:rest[1]]
            else:
                # the rest signal is provided as is
                rest_signal = rest
            rest_zero_mean = rest_signal - np.mean(rest_signal)
            statistics = signal_stats(signal=rest_zero_mean)
            var_rest = statistics['var']
        else:
            raise TypeError("Please specify the rest analysis.")
    elif isinstance(rest, dict):
        # if the input is a dictionary
        var_rest = rest['var']
    else:
        raise TypeError("Please specify the rest analysis.")






    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)
    '''
    # smoother
    # full-wave rectification
    fwlo = np.abs(signal_zero_mean)

    # smooth
    size = int(sampling_rate * 0.05)
    mvgav, _ = smoother(signal=fwlo,
                        kernel='boxzen',
                        size=size,
                        mirror=True)
    mvgav2, _ = smoother(signal=signal_zero_mean,
                         kernel='boxzen',
                         size=size,
                         mirror=True)
    '''
    tf_list = []
    onset_time_list = []
    offset_time_list = []
    alarm_time = 0
    state_duration = 0
    j = 0
    n = 0
    onset = False
    alarm = False
    print(len(signal))
    for k in range(1, len(signal), 2):  # odd values only
        # calculate the test function
        tf = (1 / var_rest) * (signal[k-1]**2 + signal[k]**2)
        tf_list.append(tf)
        if onset is True:
            if alarm is False:
                if tf < threshold2:
                    alarm_time = k // 2
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of inactive state
                if tf < threshold2:
                    state_duration += 1
                    if j > 0:  # there was one (or more) samples above the threshold level but now one is bellow it
                        # the test function may go above the threshold , but each time not longer than j samples
                        n += 1
                        if n == samples_above_fail:
                            n = 0
                            j = 0
                    if state_duration == active_state_duration_end:
                        offset_time_list.append(int((alarm_time+k//2)//2))
                        onset = False
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0
                else:  # sample falls below the threshold level
                    j += 1
                    if j > fail_size:
                        # the inactive state is above the threshold for longer than the predefined number of samples
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0
        else:  # we only look for another onset if a previous offset was detected
            if alarm is False:  # if the alarm time has not yet been identified
                if tf >= threshold1:  # alarm time
                    alarm_time = k // 2
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of active state
                if tf >= threshold1:
                    state_duration += 1
                    if j > 0:  # there was one (or more) samples below the threshold level but now one is above it.
                        # a total of n samples must be above it
                        n += 1
                        if n == samples_above_fail:
                            n = 0
                            j = 0
                    if state_duration == active_state_duration_begin:
                        onset_time_list.append(int((alarm_time+k//2)//2))
                        onset = True
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0
                else:  # sample falls below the threshold level
                    j += 1
                    if j > fail_size:
                        # the active state has fallen below the threshold for longer than the predefined number of samples
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0
    '''
    if len(offset_time_list)>0:
        onsets = np.union1d(onset_time_list[0],
                            offset_time_list[len(offset_time_list)-1])
    else:
        onsets = np.union1d(onset_time_list[0],int(len(signal)/2)-1)
    '''
    if len(offset_time_list)>0:
        onsets = np.union1d(onset_time_list,
                            offset_time_list)
    else:
        onsets = np.union1d(onset_time_list,int(len(signal)/2)-1)
    # adjust indices because of odd numbers
    onsets *= 2

    return ReturnTuple((onsets, tf_list,signal), ('onsets', 'processed','smooth'))


data_processing(data_path,0,100.0,0)
