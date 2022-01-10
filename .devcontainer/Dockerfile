FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG DIR=/workspaces/functionrl/

RUN apt-get -y update
RUN apt-get -y install git python3-opencv tmux vim
RUN apt-get -y install python3-venv
RUN pip install -U pip

# create non-root user so any created files have host user permissions
ARG USERNAME=john
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
USER $USERNAME

# install poetry for package management
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="~/.local/bin:$PATH"

WORKDIR $DIR 

# install just dependencies so they get cached
COPY pyproject.toml poetry.lock $DIR 
RUN poetry install

# install functionrl into poetry env
COPY . $DIR
RUN poetry install

# install a second poetry env that includes jupyter, and editable functionrl
RUN cd notebooks && poetry install

ENV SHELL /bin/bash