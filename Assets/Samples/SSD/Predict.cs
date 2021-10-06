using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlowLite;
using UnityEngine.UI;
using System.IO;

public class Predict : MonoBehaviour
{

    [SerializeField] string fileName = "cnn_crack_lite.tflite";

    Interpreter interpreter;

    // Start is called before the first frame update
    void Start()
    {
        var options = new InterpreterOptions();
        options.threads = SystemInfo.processorCount;

        interpreter = new Interpreter(FileUtil.LoadFile(fileName), options);
        interpreter.AllocateTensors();

        int image_size = 120;

        Texture2D myImage = LoadPNG(Directory.GetCurrentDirectory() + "/Assets/Resources/00123.jpg");
        
        var pixels = myImage.GetRawTextureData<Color32>();
        int width = myImage.width;
        int height = myImage.height - 1;
        int raw = myImage.height/ image_size; // int
        int col = myImage.width / image_size; // int
        float[,] myGrayImage = new float[height + 1, width];

        for (int i = 0; i < pixels.Length; i++)
        {
            int y = height - i / width;
            int x = i % width;
            myGrayImage[y, x] = RGBToGray(pixels[i].r, pixels[i].g, pixels[i].b);
        }

        //Debug.Log((float)myGrayImage[0, 0] / 255);

        //Debug.Log(pixels[0].r);
        //Debug.Log(pixels[0].g);
        //Debug.Log(pixels[0].b);

        //float[,,,] image_input = new float[1, image_size, image_size, 1];

        //int[,] outputs0 = new int[1,2];

        /*
        for (int r = 0; r < raw * image_size; r++)
        {
            for (int c = 0; c < col * image_size; c++)
            {
                int image_r = r / image_size;
                int inner_r = r % image_size;
                int image_c = c / image_size;
                int inner_c = c % image_size;

                //Debug.Log(image_r * col + image_c);
                //Debug.Log(inner_r);
                //Debug.Log(inner_c);
                image_input[0, inner_r, inner_c, 0] = (float)myGrayImage[r, c] / 255;
            }
        }
        */
        float[,,,] image_input = new float[1, image_size, image_size, 1];

        float[,] outputs0 = new float[1, 2];


        for (int i = 0; i < raw * col; i++)
        {
            int gr = i / col;
            int gc = i % col;

            for (int r = 0; r < image_size; r++)
            {
                for (int c = 0; c < image_size; c++)
                {
                    image_input[0, r, c, 0] = (float)myGrayImage[(gr*image_size) + r, (gc * image_size) + c] / 255;
                }
            }
            interpreter.SetInputTensorData(0, image_input);
            interpreter.Invoke();
            outputs0 = new float[1, 2];
            interpreter.GetOutputTensorData(0, outputs0);
            //Debug.Log(i);
            Debug.Log(resultArgMax(outputs0));
            Debug.Log(outputs0[0,0]);
            Debug.Log(outputs0[0,1]);
        }
    }

    void onDestroy()
    {
        interpreter?.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public float RGBToGray(float r, float g, float b)
    {
        return (float)(0.30 * r + 0.59 * g + 0.11 * b);
        //return (float)(0.30 * r + 0.50 * g + 0.20 * b);
    }

    public static Texture2D LoadPNG(string filePath)
    {

        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(2, 2);
            tex.LoadImage(fileData); //..this will auto-resize the texture dimensions.
        }
        return tex;
    }

    public int resultArgMax(float[,] result)
    {
        //Debug.Log(result[0, 0]);
        //Debug.Log(result[0, 1]);
        if (result[0, 0] > result[0, 1])
        {
            return 0;
        }
        return 1;
    }
}
