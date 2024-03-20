import os
import torch
from pathlib import Path

import dominate
from dominate.tags import a, br, h3, img, meta, p, table, td, tr

from .base import BaseVisualizer

# from ..writer.image import save_image, tensor2im

import logging
import torchvision

log = logging.getLogger(__name__)


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.
    It consists of functions such as <add_header> (add a text header to the HTML file),
    <add_images> (add a row of images to the HTML file),
    and <save> (save the HTML to the disk). It is based on Python library 'dominate',
    a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir: os.path, title: str, refresh: int = 0):
        """Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage.
            HTML file will be created at <web_dir>/index.html;
            images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, "images")
        Path(self.web_dir).mkdir(parents=True, exist_ok=True)
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self) -> os.path:
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text: str):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, images: list, texts: list, links: list, width: int = 400):
        """add images to the HTML file
        Parameters:
            images (str list) -- a list of image paths
            texts (str list) -- a list of image names shown on the website
            links (str list) -- a list of hyperref links; when you click an image,
                                it will redirect you to a new page
            width (int) -- width of the page
        """
        t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(t)
        with t:
            with tr():
                for im, txt, link in zip(images, texts, links):
                    with td(
                        style="word-wrap: break-word;", halign="center", valign="top"
                    ):
                        with p():
                            with a(href=os.path.join("images", link)):
                                img(
                                    style="width:%dpx" % width,
                                    src=os.path.join("images", im),
                                )
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HTML file"""
        html_file = "%s/index.html" % self.web_dir
        f = open(html_file, "wt")
        f.write(self.doc.render())
        f.close()


class HTMLVisualizer(BaseVisualizer):
    """
    This class includes several functions that can display/save images to html.
    It uses a Python library 'dominate' (wrapped in 'HTML')
        for creating HTML files with images.
    """

    def __init__(self, instance_config, visualizer_config):
        """Initialize the HTML visualizer class
        Parameters:
            opt -- stores all the experiment configs
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saving HTML filters
        Step 4: create a logging file to store training losses
        """
        super().__init__(instance_config, visualizer_config)

        self.name = self.instance_config["name"]
        self.win_size = self.instance_config["display_winsize"]

        self.instance_root_dir = os.path.join(
            self.visualizer_config["root_dir"], self.name
        )
        self.img_dir = os.path.join(self.instance_root_dir, "images")
        log.info("create web directory %s..." % self.instance_root_dir)
        for path in [self.instance_root_dir, self.img_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

        self.saved_step_list = []

    def add_scalar(
        self, tag: str, scalar_value: float, step: str = None, wall_time=None
    ):
        pass

    def add_scalars(
        self, tag: str, scalar_dict: dict, step: str = None, wall_time=None
    ):
        pass

    def add_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: str,
        bins: str = "tensorflow",
        wall_time=None,
    ):
        pass

    def add_image(
        self, tag: str, img_tensor: torch.Tensor, step: str = None, wall_time=None
    ):
        pass

    def add_images(  # for batch images
        self, tag: str, img_tensor: torch.Tensor, step: str = None, wall_time=None
    ):
        pass

    def add_image_dict(
        self, tag: str, img_tensor_dict: dict, step: str = None, wall_time=None
    ):
        # Can support batch visualization
        self.saved_step_list.append(step)

        for label, image in img_tensor_dict.items():
            # image_numpy = tensor2im(image)
            img_path = os.path.join(self.img_dir, f"step_{step}_label_{label}.png")
            torchvision.utils.save_image(image, img_path)

        # update website
        webpage = HTML(
            self.instance_root_dir, "Experiment name = %s" % self.name, refresh=0
        )
        for step in self.saved_step_list:
            webpage.add_header(f"Step: {step}")
            images, texts, links = [], [], []
            for label, _ in img_tensor_dict.items():
                img_path = f"step_{step}_label_{label}.png"
                images.append(img_path)
                texts.append(label)
                links.append(img_path)
            webpage.add_images(images, texts, links, width=self.win_size)
        webpage.save()

    def add_video(
        self,
        tag: str,
        vid_tensor: torch.Tensor,
        fps: int = None,
        step: str = None,
        wall_time=None,
    ):
        pass

    def add_graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor,
        verbose: bool = False,
    ):
        pass

    def add_embedding(
        self,
        mat: torch.Tensor,
        metadata: list,
        label_img: torch.Tensor,
        step: str,
        tag: str,
    ):
        pass

    def add_hparams_summary(
        self, hparam_dict: dict, metric_dict: dict, run_name: str = None
    ):
        pass

    def close(self):
        # TODO: implement close in the training code
        pass
