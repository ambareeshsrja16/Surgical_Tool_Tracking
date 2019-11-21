#####################
# DLC Dockerfile from https://github.com/MMathisLab/Docker4DeepLabCut2.0/blob/master/Dockerfile
#####################

FROM python:3

RUN pip install imageio
# install ffmpeg from imageio.
RUN pip install imageio-ffmpeg

FROM bethgelab/deeplearning:cuda9.0-cudnn7
RUN apt-get update
RUN apt-get -y install ffmpeg

RUN pip install --upgrade pip

RUN pip install tensorflow-gpu==1.8

RUN pip3 install deeplabcut

RUN pip install ipywidgets
RUN pip3 install ipywidgets

RUN pip3 install seaborn



#####################
# CREDITS : JJJ
#####################


# install ROS-core and base packages
RUN apt-get update && apt-get install -q -y \
  dirmngr \
  gnupg2 \
  lsb-release \
  && rm -rf /var/lib/apt/lists/*


# setup keys
# Get correct key : https://hub.docker.com/_/ros?tab=description
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  python-rosdep \
  python-rosinstall \
  python-vcstools \
  && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
  && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
  RUN apt-get update && apt-get install -y \
  ros-kinetic-ros-core=1.3.2-0* \
  && rm -rf /var/lib/apt/lists/*

# install ros base packages
RUN apt-get update && apt-get install -y \
  ros-kinetic-ros-base=1.3.2-0* \
  && rm -rf /var/lib/apt/lists/*

# install the robot packages
RUN apt-get update && apt-get install -y \
  ros-kinetic-robot=1.3.2-0* \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/DLCROS_ws/src

WORKDIR /root/DLCROS_ws/src

RUN git clone https://github.com/ambareeshsrja16/Surgical_Tool_Tracking.git

# getting the catkin build tools 
RUN apt-get update && apt-get install -y \
  python-catkin-pkg\
  python-catkin-tools\
  && rm -rf /var/lib/apt/lists/*

# Important packages for building the messages in python 3
RUN apt-get update && apt-get install -y \
  ros-kinetic-catkin \
  python3-empy \
  python3-catkin-pkg-modules \
  python3-rospkg-modules\
  && rm -rf /var/lib/apt/lists/*


# Python packages for gazebo-ros-commmunication
RUN apt-get update && apt-get install -y \
  ros-kinetic-gazebo-ros-pkgs \
  ros-kinetic-gazebo-ros-control \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-dev python3-numpy python3-yaml ros-kinetic-cv-bridge

WORKDIR /root/DLCROS_ws

# Clone cv_bridge src | Clone the specific branch from the repo to avoid the "Can't find Boost libraries" Refer: https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3?rq=1
RUN git clone -b branch1.12.8 https://github.com/ambareeshsrja16/vision_opencv.git src/vision_opencv

WORKDIR /root/DLCROS_ws/

# Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
RUN catkin config --install
RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash && \
  catkin build --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so"

# Modify the ~/.bashrc by adding the necessary bash files to it, now when the kernerl opens with bash, it would have sourced the necessary stuff
RUN /bin/bash -c "echo 'source /opt/ros/kinetic/setup.bash' >> ~/.bashrc"
RUN /bin/bash -c "echo 'source ~/DLCROS_ws/install/setup.bash --extend' >> ~/.bashrc"

CMD ["bash"]
