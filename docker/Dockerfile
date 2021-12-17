FROM arm64v8/ubuntu:18.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install git python-pip gcc gfortran perl python-dev libblas-dev liblapack-dev g++ libatlas-base-dev libfreetype6-dev pkg-config -y
RUN export DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/America/Chicage /etc/localtime
RUN apt-get install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev  libharfbuzz-dev libfribidi-dev libxcb1-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install wheel==0.36.2

# install B_net
RUN git clone https://github.com/max2022/B_net.git
RUN cd B_net
RUN pip install -r requirements.txt

# compile and install scikit image 0.14.5
RUN git clone https://github.com/tutu-kang/scikit-image-arm64v8.git
RUN cd scikit-image-arm64v8
RUN python setup build_ext -i
RUN python -m pip install -e .
RUN cd ..

#todo, modify lenet_script and only read test data to save loading time. 
RUN python lenet_script.py
