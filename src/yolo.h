#ifndef YOLO_H
#define YOLO_H

void execute_yolo_model(char *filename, float thresh);
void load_yolo_model(char *cfgfile, char *weightfile);
#endif
