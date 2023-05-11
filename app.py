# Written by Tim de Wild, May 2023, University of Groningen, the Netherlands

import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import erf
from scipy.stats import norm

#--- Function definitions go here ---#

def prob(z1,z2):
    return 0.5*(erf(z2/np.sqrt(2)) - erf(z1/np.sqrt(2)))


def prob_LT(z):
    return prob(-np.inf, z)


def prob_RT(z):
    return prob(z, +np.inf)


def prob_TT(z):
    return 2*prob_RT(z)


def prob_aTT(z1, z2):
    return prob_LT(z1) + prob_RT(z2)


def prob_S(z):
    return prob(-z,z)

@st.cache_data()
def ptype_label(n):
    labels = {
        1: 'Left-tailed',
        2: 'Right-tailed',
        3: 'Symmetric two-tailed',
        4: 'Assymetric two-tailed',
        5: 'Symmetric bulk',
        6: 'Assymmetric bulk'
    }
    return labels[n]

@st.cache_data()
def p_math_label(n):
    labels = {
        1: 'P[X\leq x]',
        2: 'P[X\geq x]',
        3: 'P[|X|\leq x]',
        4: 'P[X\leq x_1 \wedge X\geq x_2]',
        5: 'P[|X|\leq x]',
        6: 'P[x_1\leq X \leq x_2]'
    }
    return labels[n]

@st.cache_data()
def image_selector(n):
    images = {
        1: './images/left_tailed.png',
        2: './images/right_tailed.png',
        3: './images/two_tailed_symmetric.png',
        4: './images/two_tailed_assymmetric.png',
        5: './images/bulk_symmetric.png',
        6: './images/bulk_assymmetric.png'
    }
    return images[n]

@st.cache_data()
def ptype_info(n):
    labels = {
        1: "Integration bound $x$ can have any value.",
        2: "Integration bound $x$ can have any value.",
        3: "Integration bound $x$ of the right tail should be given and satisfy $x\geq \mu$, the symmetric left tail bound will be calculated automatically.",
        4: "The integration bounds must satisfy $x_1\leq \mu$ and $x_2\geq \mu$.",
        5: "The right integration bound should be given and satisfy $x\geq \mu$. The left bound will be calculated automatically.",
        6: "The bounds must satisfy $x_2>x_1$."
    }
    return labels[n]

@st.cache_data()
def load_images():
    images = []
    for i in [1,2,3,4,5,6]:
        img = Image.open(image_selector(ptype))
        images.append(img)
    return images

@st.cache_data()
def plot_gauss(mu, sigma):
    ymax = 1/np.sqrt(2*np.pi*sigma**2)

    x = np.linspace(mu-4*sigma, mu+4*sigma, 1000)
    y = norm.pdf(x, loc = mu, scale = sigma)

    data = [x, y]

    fig, ax = plt.subplots(figsize=(8,5))

    ax.grid(True,color=u'white',lw=1.5)
    ax.set_facecolor('#EEEEEE')

    ax.plot(x,y)
    ax.set_xlim(mu-4*sigma, mu+4*sigma)
    ax.set_ylim(0,1.2*ymax)

    ax.set_title(f"Gaussian distribution with $\mu={mu}$ and $\sigma={sigma}$.")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")

    return fig, ax, data

def add_area(ax, data, x1, x2):
    x, y = data

    ax.fill_between(x, y, 0,
                 where = (x > x1) & (x <= x2),
                 color = 'tab:blue',
                 alpha = 0.5, 
                 zorder = 3
                 )
    
def add_prob_label(ax, prob):
    ax.plot([],[], c='tab:blue', lw = 8, alpha=0.5, label=f"$P={round(prob, 3)}$")
    ax.legend()

def x1_format(ptype, mu, sigma, t):
    # t should be 'value', 'min' or 'max'
    formats = {
            1: {'value': mu - sigma,    'min': None,    'max': None},
            2: {'value': mu + sigma,    'min': None,    'max': None},
            3: {'value': mu + 2*sigma,  'min': mu,      'max': None},
            4: {'value': mu - sigma,    'min': None,    'max': mu},
            5: {'value': mu + sigma,    'min': mu,      'max': None},
            6: {'value': mu - sigma,    'min': None,    'max': None}
        }
    
    return formats[ptype][t]

def x2_format(ptype, mu, sigma, t, x1):
    formats = {
            4: {'value': mu + 2*sigma,  'min': mu,      'max': None},
            6: {'value': mu + 2*sigma,  'min': x1,    'max': None}
        }
    
    return formats[ptype][t]


#--- App starts here ---#
st.write("""   
    # Gaussian Probability Calculator  
""")

with st.sidebar:
    st.write("""
    # Control Panel
    ## Probability Type
    """)
    ptype = st.radio('Select your probability type here:', [1,2,3,4,5,6], 0, format_func = ptype_label )

    st.write("""
    ## Visual Preview
    The blue area indicates the probability that will be calculated.
    """)

    st.image(
        image = Image.open(image_selector(ptype)),
        )
    
    st.info(ptype_info(ptype) , icon = "ℹ️")
    



st.write("""
    #### Specifying the Gaussian
    Specify the Gaussian distribution here via its mean $\mu$ and standard deviation $x$. 
""")

c1, c2 = st.columns(2)

with c1: 
    mu = st.number_input("mean $\mu$", value=0.0, format="%f")

with c2: 
    sigma  = st.number_input("standard deviation $\sigma$", value=1.0, min_value=1.0e-15, format="%f")

_, c_fig1, _ = st.columns([1,6,1])

fig1, _, _ = plot_gauss(mu,sigma)

with c_fig1:
    st.pyplot(fig1)

st.write("""
    #### Integration Bound(s)
    Specify the integration bound(s) here below. 
    """)

c3, c4 = st.columns(2)

if ptype in [1,2,3,5]:
    with c3:
        x = st.number_input("integration bound $x$", 
                            min_value = x1_format(ptype, mu, sigma, 'min'), 
                            max_value = x1_format(ptype, mu, sigma, 'max'), 
                            value = x1_format(ptype, mu, sigma, 'value'),  
                            format="%f"
                            )
        
        z = (x - mu)/sigma

        if np.abs(x-mu) > 4*sigma:
            st.info("The integration bound is more than $\pm 4\sigma$ away from the mean $\mu$, the integration area might not visible.", icon="ℹ️")

if ptype not in [1,2,3,5]:
    with c3:
        x1 = st.number_input("left integration bound $x_1$",
                             min_value = x1_format(ptype, mu, sigma, 'min'), 
                             max_value = x1_format(ptype, mu, sigma, 'max'), 
                             value = x1_format(ptype, mu, sigma, 'value'),  
                             format="%f"
                            )
        
        z1 = (x1 - mu)/sigma
        
        if np.abs(x1-mu) > 4*sigma:
            st.info("The integration bound is more than $\pm 4\sigma$ away from the mean $\mu$, the integration area might not visible.", icon="ℹ️")
    with c4: 
        x2 = st.number_input("right integration bound $x_2$",
                             min_value = x2_format(ptype, mu, sigma, 'min', x1), 
                             max_value = x2_format(ptype, mu, sigma, 'max', x1), 
                             value = x2_format(ptype, mu, sigma, 'value', x1),  
                             format="%f"
                            )
        
        z2 = (x2 - mu)/sigma
        
        if np.abs(x2-mu) > 4*sigma:
            st.info("The integration bound is more than $\pm 4\sigma$ away from the mean $\mu$, the integration area might not visible.", icon="ℹ️")

fig2, ax2, data = plot_gauss(mu, sigma)

if ptype == 1:
    add_area(ax2, data, -np.inf, x)
    P = prob_LT(z)

if ptype == 2: 
    add_area(ax2, data, x, +np.inf)
    P = prob_RT(z)

if ptype == 3:
    add_area(ax2, data, -np.inf, 2*mu-x)
    add_area(ax2, data, x, +np.inf)
    P = prob_TT(z)

if ptype == 4:
    add_area(ax2, data, -np.inf, x1)
    add_area(ax2, data, x2, +np.inf)
    P = prob_aTT(z1,z2)

if ptype == 5:
    add_area(ax2, data, 2*mu-x, x)
    P = prob_S(z)

if ptype == 6:
    add_area(ax2, data, x1, x2)
    P = prob(z1, z2)

add_prob_label(ax2, P)

_, c_fig2, _ = st.columns([1,6,1])

with c_fig2:
    st.pyplot(fig2)

fig2.clf()

st.write(f"""
    #### Result
    The calculated probability is ${p_math_label(ptype)} < {round(P,3)}$. 
""")













