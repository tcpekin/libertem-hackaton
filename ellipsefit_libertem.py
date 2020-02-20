import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings("ignore")

from libertem.udf.base import UDF
from libertem import api
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.hdf5 import _get_datasets
from libertem.udf.raw import PickUDF

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim

from py4DSTEM.process.utils.ellipticalCoords import fit_double_sided_gaussian
from py4DSTEM.process.utils.ellipticalCoords import compare_double_sided_gaussian

plt.ion()


class MeanPatternUDF(UDF):
    def get_result_buffers(self):
        return {
            "mean_dp": self.buffer(kind="sig", dtype=np.float32),
            "frame_count": self.buffer(kind="single", dtype=np.int64),
            "vbf": self.buffer(kind="nav", dtype=np.float32),
        }

    def process_frame(self, frame):
        self.results.mean_dp[:] = self.results.mean_dp + frame
        self.results.frame_count[:] += 1
        self.results.vbf[:] = np.sum(frame)

    def merge(self, dest, src):
        # if a merge function exists, you must merge all your results
        # if a merge function does not exist, it will automatically put them there for buffers of type 'nav'
        # merge functions are always needed for types single and sig.
        dest["mean_dp"][:] += src["mean_dp"]
        dest["frame_count"][:] += src["frame_count"]
        dest["vbf"][:] = src["vbf"]


class EllipseUDF(UDF):
    def __init__(self, p0, mask=None):
        super().__init__(
            mask=mask, p0=p0
        )  # the arguments must get passed to the super UDF class such that the workers and controller all have the same data

    def get_result_buffers(self):
        return {
            "ellipse_coefs": self.buffer(
                kind="nav", extra_shape=(12,), dtype=np.float32
            )
        }

    def process_frame(self, frame):
        mask = (
            self.params.mask
        )  # this is where all keyword arguments are put in the super class
        p0 = self.params.p0
        self.results.ellipse_coefs[:] = fit_double_sided_gaussian(frame, p0, mask=mask)


if __name__ == "__main__":
    with api.Context() as ctx:
        # debug_ctx = api.Context(executor=InlineJobExecutor())  # for debug purposes

        path = "/Users/Tom/Documents/Research/code/others/libertem-hackaton/christoph_pillar_128x65x224x240.h5"
        ds_path = _get_datasets(path)[0][0]

        ds = ctx.load("hdf5", ds_path=ds_path, path=path)

        # this is how you get a single frame
        temp_im = ctx.create_pick_analysis(dataset=ds, x=40, y=40)
        res = ctx.run(temp_im)
        im = res.intensity.raw_data

        # this is how you get a mean image - the important part is that the final combination is local - you do not compute the mean on the workers, it makes no sense
        udf_mean = MeanPatternUDF()
        mean_data = ctx.run_udf(udf=udf_mean, dataset=ds, progress=True)
        mean_im = mean_data["mean_dp"].data / mean_data["frame_count"].data
        virtual_brightfield = mean_data["vbf"].data

        # we want to ignore the vacuum
        roi_pillar = virtual_brightfield > 1e8

        # for testing
        # roi_pillar = np.zeros_like(roi_pillar)
        # roi_pillar[:, 30] = True

        # we have a beam stop
        mask_haadf = spim.median_filter(
            np.logical_and(mean_im > 2000, mean_im < 4e3), 5
        )
        yy, xx = np.meshgrid(
            np.arange(mask_haadf.shape[1]), np.arange(mask_haadf.shape[0])
        )
        yy, xx = yy - 124, xx - 112  # centering
        radius = np.sqrt(xx ** 2 + yy ** 2)
        mask_r = np.logical_and(radius > 50, radius < 80)
        mask = mask_haadf * mask_r

        # initial coefficients
        # init_coef = fit_double_sided_gaussian(
        #     mean_im, [4e2, 4e3, 50, 10, 10, 2e3, 60, 115, 120, 1, 0, 1], mask=mask
        # )

        init_coef = [4e3, 3e3, 10, 7, 7, 3e3, 65, 116, 124, 1, 0, 1]

        # fit all of the patterns in the ROI
        udf_ellipse = EllipseUDF(mask=mask, p0=init_coef)
        # roi = np.zeros(ds.shape.nav, bool)
        # roi[125] = True
        # result = ctx.run_udf(udf=udf_ellipse, dataset=ds, roi=roi)
        result = ctx.run_udf(udf=udf_ellipse, dataset=ds, roi=roi_pillar, progress=True)

        fit = result["ellipse_coefs"].raw_data

        # also get diffraction patterns for this same roi - only for testing
        # udf_pick = PickUDF()
        # stack = ctx.run_udf(udf=udf_pick, dataset=ds, roi=roi_pillar)
        # stack = stack["intensity"].raw_data

    # if not using with statement, run ctx.close() to kill workers
