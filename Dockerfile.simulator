FROM ubuntu:noble
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -yq install python3-pip python3-venv
COPY simulator.py /simulator/
COPY simulator_test.py /simulator/
WORKDIR /simulator
RUN python3 -m venv /simulator
RUN /simulator/bin/python3 simulator_test.py
COPY messages.mllp /data/
EXPOSE 8440
EXPOSE 8441
ENTRYPOINT ["/simulator/bin/python3", "/simulator/simulator.py"]
CMD ["--messages=/data/messages.mllp"]
