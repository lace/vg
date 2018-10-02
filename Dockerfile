# This Dockerfile is used to create a static build on Zeit Now for previewing
# documentation in pull requests.
# https://zeit.co/docs/static-deployments/builds/building-with-now

FROM python:3.6
COPY requirements*.txt /tmp/
RUN pip install --upgrade -r /tmp/requirements_dev_py3.txt
RUN rm /tmp/requirements*.txt
WORKDIR /src
COPY . /src
RUN ./dev.py doc
RUN cp -r doc/build /public
