#ifndef YOLO_H
#define YOLO_H
#include "image.h"
//void execute_yolo_model(image im, float thresh);
void execute_yolo_model(image im, float thresh,box *boxes,float **probs);
int load_yolo_model(char *cfgfile, char *weightfile);
#endif
