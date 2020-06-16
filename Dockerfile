
FROM continuumio/miniconda3


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN mkdir -p /usr/share/man/man1mkdir -p /usr/share/man/man1
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
    openjdk-11-jre-headless

RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n fever python=3.6


RUN mkdir /fever/
RUN mkdir /work

VOLUME /work
WORKDIR /fever

RUN conda install -c pytorch pytorch=0.4.1 -y

RUN mkdir /fever/data

WORKDIR /fever/data
RUN wget -O "aux_file.zip" "https://www.dropbox.com/s/yrecf582rqtgke0/aux_file.zip?dl=0"
RUN unzip "aux_file.zip" && rm "aux_file.zip"

RUN mkdir /fever/dep_packages
WORKDIR /fever/dep_packages
RUN wget -O "dep_packages.zip" "https://www.dropbox.com/s/74uc24un1eoqwch/dep_packages.zip?dl=0"
RUN unzip "dep_packages.zip" && rm "dep_packages.zip"

RUN mkdir /fever/data/models
WORKDIR /fever/data/models
RUN wget -O "saved_nli_m.zip" "https://www.dropbox.com/s/rc3zbq8cefhcckg/saved_nli_m.zip?dl=0"
RUN unzip "saved_nli_m.zip" && rm "saved_nli_m.zip"
RUN  wget -O "nn_doc_selector.zip" "https://www.dropbox.com/s/hj4zv3k5lzek9yr/nn_doc_selector.zip?dl=0"
RUN unzip "nn_doc_selector.zip" && rm "nn_doc_selector.zip"

RUN wget -O "saved_sselector.zip" "https://www.dropbox.com/s/56tadhfti1zolnz/saved_sselector.zip?dl=0"
RUN unzip "saved_sselector.zip" && rm "saved_sselector.zip"

WORKDIR /fever/data
RUN wget -O "chaonan99.zip" "https://www.dropbox.com/s/pu3h5xc2kpws0n2/chaonan99.zip?dl=0"
RUN unzip "chaonan99.zip" && rm "chaonan99.zip"

WORKDIR /fever
RUN mv data/chaonan99/* data/
RUN mv "data/models/saved_sselector/i(57167)_epoch(6)_(tra_score:0.8850885088508851|raw_acc:1.0|pr:0.3834395939593578|rec:0.8276327632763276|f1:0.5240763176570098)_epoch" data/models/sent_selector
RUN mv "data/models/saved_sselector/i(58915)_epoch(7)_(tra_score:0.8838383838383839|raw_acc:1.0|pr:0.39771352135209675|rec:0.8257575757575758|f1:0.5368577222846761)_epoch" data/models/sent_selector_1
RUN mv "data/models/saved_sselector/i(77083)_epoch(7)_(tra_score:0.8841384138413841|raw_acc:1.0|pr:0.3964771477147341|rec:0.8262076207620762|f1:0.5358248492912955)_epoch" data/models/sent_selector_2
RUN mv data/models/nn_doc_selector data/models/nnds
RUN mv "data/models/nnds/i(9000)_epoch(1)_(tra_score:0.9212421242124212|pr:0.4299679967996279|rec:0.8818631863186318|f1:0.5780819247968391)" data/models/nn_doc_selector
RUN mv "data/models/saved_nli_m/i(77000)_epoch(11)_dev(0.6601160116011601)_loss(1.1138329989302813)_seed(12)" data/models/nli
RUN rm -rf data/models/saved_sselector
RUN rm -rf data/models/nnds
RUN rm -rf data/models/saved_nli_m


WORKDIR /fever/
COPY --from=feverai/common /local/fever-common/data/wiki-pages /tmp/wiki-pages
ADD requirements.txt /fever/
RUN pip install -r requirements.txt

ADD src /fever/src/
ADD scripts /fever/scripts/

ENV PYTHONPATH src
ENV CLASSPATH=/fever/dep_packages/stanford-corenlp-full-2017-06-09/*

RUN python -c 'import nltk; nltk.download("wordnet_ic"); nltk.download("averaged_perceptron_tagger"); nltk.download("wordnet")'
RUN python src/utils/build_db.py

CMD ["waitress-serve", "--host=0.0.0.0","--port=5000", "--call", "app:web"]
