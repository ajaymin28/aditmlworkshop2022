FROM python:3.7
WORKDIR /
COPY . .
RUN apt-get update
RUN apt-get install ca-certificates -y
RUN pip3 install --upgrade setuptools pip
RUN pip3 install wheel
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["sh","./startup.sh"]