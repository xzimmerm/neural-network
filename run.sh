#!/bin/bash

cd src
module add jdk-21
javac -cp . nn/demo/main.java
java nn.demo.Main