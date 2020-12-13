# Vehicle detection web application using Django and AWS

## Introduction: 
This project focus on deploying end-to-end vehicle detection web application using below mentioned programming language, algorithm, frameworks and tools. 

## Prerequisites: 
Programming language: Python 

Deep learning algorithm: YOLO V3 

AWS Service: Amazon Elastic Compute Cloud (Amazon EC2), Amazon Elastic Block Store (EBS) 

Web Framework: Django 

Libraries: OpenCV, Numpy 

Server: NGINX 

Web Server Gateway Interface (WSGI) used: Gunicorn 

Tools: PuTTYgen , PuTTY and MobaXterm 

## Project folder structure: 


- media :Contains all uploaded and classified video files 

- models : Contains 3 files: yolov3 weight file,yolov3 configuration file and coco dataset file 

- templates : Contains html file (web application interface) 

- video : Django application file 

- videotest : Django project file 

## Installation:

1. Create Amazon EC2 instance. 

2. Connect created EC2 instance with your local machine using private SSH key using PuTTY or MobaXterm 

3. Install python, OpenCV, Numpy, django 

    sudo apt-get install python3.6 

    pip install opencv-python 

    pip install numpy 

4. Clone your project from github to your EC2 instance 

    git clone project_url 

5. Install Nginx server 

    sudo apt-get install –y nginx 

6. Install gunicorn 

    pip install gunicorn 

7. Bind project wsgi file to the nginx server using gunicorn through port 8000 

    gunicorn –bind 0.0.0.0:8000 app_name.wsgi:application 

8. Test if the project in running in nginx server in AWS EC2 instance on your browser.For example

    public_DNS(of EC2 instance):8000

 

Note before deploying the project we must set following in django project setting file. 

 DEBUG = False 

 ALLOWED_HOSTS = ['*'] 

And edit nginx configuration file to access the media folder of the project in EC2 instance. 


## Web application architecture:   

Web application template is built with basic html and CSS (JavaScript are better suited for front-end development but since our web interface is very simple using html and CSS is enough). The html form is used to input the image and the videos which needs to be classified. The uploaded image/videos are then saved in local drive in case of localhost as a server and in EBS volumes in-case Nginx with AWS is used as server. saved image/video is then sent to the yolo algorithm for classification which is again saved and then displayed in the web page. 


![](web_application/zimage_for_readme/image1.jpg)

                                          Fig 1: web application architecture 

## web application interface: 
The web interface consists of one simple form to upload the video to be classified and two videos output sections: one of which is the original uploaded video and second one is the classified output video. 

![](web_application/zimage_for_readme/image3.JPG)

                                          Fig 2: web application interface 



# Author: Ananta Khanal                                           




