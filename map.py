import argparse
import cv2
import numpy as np
import time

start_time = time.time()

def find_harris_corners(input_img, k, window_size, threshold):
    
    corner_list = []
    output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)
    
    offset = int(window_size/2)
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset
    
    
    dy, dx = np.gradient(input_img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    
    
    for y in range(offset, y_range):
        for x in range(offset, x_range):
            
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            r = det - k*(trace**2)

            if r > threshold:
                corner_list.append([x, y, r])
                output_img[y,x] = (0,0,255)
    
    return corner_list, output_img 

def main():
    
    
    k = 0.04
    window_size = 5
    threshold = 10000.00
    
        
    input_img = cv2.imread("ex1.png", 0)
    
    cv2.imshow('Исходное изображение', input_img)
    #cv2.waitKey(0)

    if input_img is not None:
        
        print ("Начат поиск углов!")
        corner_list, corner_img = find_harris_corners(input_img, k, window_size, threshold)
        
        cv2.imshow('Найденные углы', corner_img)
        #cv2.waitKey(0)
        
        corner_file = open('corners_list.txt', 'w')
        corner_file.write('x ,\t y, \t r \n')
        for i in range(len(corner_list)):
            corner_file.write(str(corner_list[i][0]) + ' , ' + str(corner_list[i][1]) + ' , ' + str(corner_list[i][2]) + '\n')
        corner_file.close()
        
        if corner_img is not None:
            cv2.imwrite("corners_img.png", corner_img)
    else:
        print ("Ошибка!")
            
    print ("Поиск углов завершен!")
    print("Затраченное время на выполнения алгоритма собственной реализации: %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    main()
