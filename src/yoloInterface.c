#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "cv.h"
#endif

//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
//image voc_labels[20];
network glo_net;

int load_yolo_model(char *cfgfile, char *weightfile)
{
    int maxSize = 0;
    printf("%s",cfgfile);
    glo_net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&glo_net, weightfile);
    }
    set_batch_network(&glo_net, 1);
    detection_layer l = glo_net.layers[glo_net.n-1];
    return l.side*l.side*l.n; // returns the maximum possible number of detections
}

/*void execute_yolo_model(image im, float thresh) //, *float boxesOut, *float probsOut
{
    clock_t time1,time2;
    time1=clock();
    detection_layer l = glo_net.layers[glo_net.n-1];
    srand(2222222);
    
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    image sized = resize_image(im, glo_net.w, glo_net.h);
    float *X = sized.data;
    time2=clock();
    float *predictions = network_predict(glo_net, X);
    printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time1));
    printf("Not related to network %f seconds.\n", sec(time2-time1));
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
    if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
    show_image(im, "predictions");
    free_image(im);
#ifdef OPENCV
    cvWaitKey(1);
#endif

}*/

void execute_yolo_model(image im, float thresh,box *boxes,float **probs) //, *float boxesOut, *float probsOut
{
    clock_t time1,time2;
    time1=clock();
    detection_layer l = glo_net.layers[glo_net.n-1];
    srand(2222222);
    
    int j;
    float nms=.5;
    //box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    //float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    //
    image sized = resize_image(im, glo_net.w, glo_net.h);
    float *X = sized.data;
    time2=clock();
    float *predictions = network_predict(glo_net, X);
    printf("Predicted in %f seconds.\n", sec(clock()-time1));
    printf("Not related to network %f seconds.\n", sec(time2-time1));
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
    if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    free_image(sized);
}

/*void execute_yolo_model_file(char *filename, float thresh) //, *float boxesOut, *float probsOut
{
    detection_layer l = glo_net.layers[glo_net.n-1];
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, glo_net.w, glo_net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(glo_net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
        show_image(im, "predictions");
        save_image(im, "predictions");

        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}*/
