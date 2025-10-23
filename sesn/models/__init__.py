# MNIST
from .mnist_ss import mnist_ss_28, mnist_ss_56
from .mnist_sevf import mnist_sevf_scalar_28, mnist_sevf_scalar_56, mnist_sevf_vector_28, mnist_sevf_vector_56
from .mnist_cnn import mnist_cnn_28, mnist_cnn_56
from .mnist_kanazawa import mnist_kanazawa_28, mnist_kanazawa_56
from .mnist_xu import mnist_xu_28, mnist_xu_56
from .mnist_dss import mnist_dss_vector_28, mnist_dss_vector_56, mnist_dss_scalar_28, mnist_dss_scalar_56
from .mnist_ses import mnist_ses_scalar_28, mnist_ses_scalar_56, mnist_ses_vector_28, mnist_ses_vector_56
from .mnist_ses import mnist_ses_scalar_28p, mnist_ses_scalar_56p, mnist_ses_vector_28p, mnist_ses_vector_56p

from .mnist_ses_rst import mnist_ses_rst_vector_56_rot_8_interrot_8 #mnist_ses_rst_scalar_28p, mnist_ses_rst_scalar_56p, mnist_ses_rst_vector_28p, mnist_ses_rst_vector_56p
from .mnist_resnet_rst_mcg_e import resnext50_32x4d_rst_mcg_e,resnext101_32x8d_rst_mcg_e,wide_resnet50_2_rst_mcg_e,resnext26_32x4d_rst_mcg_e

from .stl_resnet_rst_mcg_a import resnext50s_32x4d_rst_mcg_a
