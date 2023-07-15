#!/bin/bash
fileid=14-uaeMAihLdMepfZAth19T1pfZIoMcaE
filename="DAL-HERS/pretrained/DAL_loss=bce-rgb_date=23Feb2021.tar"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}