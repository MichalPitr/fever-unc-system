FROM continuumio/miniconda3

ENTRYPOINT ["/bin/bash"]

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN apt-get install -y --no-install-recommends --allow-unauthenticated \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip \
    libffi-dev \
    software-properties-common \
    openjdk-8-jre-headless

RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n fever python=3.6


RUN mkdir /fever/
RUN mkdir /work

VOLUME /work


WORKDIR /fever

RUN conda install -c pytorch pytorch=0.4.1 -y
ADD requirements.txt /fever/
RUN pip install -r requirements.txt
ADD src /fever/src/
ADD scripts /fever/scripts/

RUN mkdir /fever/data

RUN mkdir /fever/data/fever
WORKDIR /fever/data/fever
RUN wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
RUN wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
RUN wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl

WORKDIR /fever/data
RUN wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
RUN unzip "wiki-pages.zip" && rm "wiki-pages.zip"

WORKDIR /fever/data
RUN wget -O "aux_file.zip" "https://www.dropbox.com/s/yrecf582rqtgke0/aux_file.zip?dl=0"
RUN unzip "aux_file.zip" && rm "aux_file.zip"

RUN mkdir /fever/dep_packages
WORKDIR /fever/dep_packages
RUN wget -O "dep_packages.zip" "https://www.dropbox.com/s/74uc24un1eoqwch/dep_packages.zip?dl=0"
RUN unzip "dep_packages.zip" && rm "dep_packages.zip"

RUN mkdir /fever/saved_models
WORKDIR /fever/saved_models
RUN wget -O "saved_nli_m.zip" "https://www.dropbox.com/s/rc3zbq8cefhcckg/saved_nli_m.zip?dl=0"
RUN unzip "saved_nli_m.zip" && rm "saved_nli_m.zip"
RUN  wget -O "nn_doc_selector.zip" "https://www.dropbox.com/s/hj4zv3k5lzek9yr/nn_doc_selector.zip?dl=0"
RUN unzip "nn_doc_selector.zip" && rm "nn_doc_selector.zip"

RUN wget -O "saved_sselector.zip" "https://www.dropbox.com/s/56tadhfti1zolnz/saved_sselector.zip?dl=0"
RUN unzip "saved_sselector.zip" && rm "saved_sselector.zip"

WORKDIR /fever/

RUN mkdir /fever/results
WORKDIR /fever/results

RUN wget -O "chaonan99.zip" "https://www.dropbox.com/s/pu3h5xc2kpws0n2/chaonan99.zip?dl=0"
RUN unzip "chaonan99.zip" && rm "chaonan99.zip"

ENV PYTHONPATH src
WORKDIR /fever

RUN python src/pipeline/prepare_data.py build_database
RUN python -c 'import nltk; nltk.download("wordnet_ic"); nltk.download("averaged_perceptron_tagger"); nltk.download("wordnet")'

ADD run.sh /fever/
