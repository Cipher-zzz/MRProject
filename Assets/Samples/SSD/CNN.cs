using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace TensorFlowLite
{
    /// <summary>
    /// Crack Detection
    /// </summary>
    public class CNN : BaseImagePredictor<float>
    {

        int[] frameOutput = null;
        int image_size = 120;
        int raw = 0;
        int col = 0;

        public readonly struct Result
        {
            public readonly int classID;
            public readonly float score;
            public readonly Rect rect;

            public Result(int classID, float score, Rect rect)
            {
                this.classID = classID;
                this.score = score;
                this.rect = rect;
            }
        }

        public CNN(string modelPath) : base(modelPath, true)
        {
        }

        public override void Invoke(Texture inputTex)
        {
            TextureResizer.ResizeOptions cnnResizeOptions = new TextureResizer.ResizeOptions()
            {
                aspectMode = AspectMode.Fill,
                rotationDegree = 0,
                mirrorHorizontal = false,
                mirrorVertical = false,
                width = inputTex.width,
                height = inputTex.height,
            };
            RenderTexture tex = resizer.Resize(inputTex, cnnResizeOptions);

            //Texture2D myImage = TextureToTexture2D(inputTex);
            Texture2D myImage = RTtoTexture2D(tex);
            var pixels = myImage.GetRawTextureData<Color32>();
 
            int width = myImage.width;
            int height = myImage.height - 1;

            raw = myImage.height / image_size; // int
            col = myImage.width / image_size; // int
            float[,] myGrayImage = new float[height + 1, width];

            for (int i = 0; i < pixels.Length; i++)
            {
                int y = height - i / width;
                int x = i % width;
                myGrayImage[y, x] = RGBToGray(pixels[i].r, pixels[i].g, pixels[i].b);
            }

            float[,,,] image_input = new float[1, image_size, image_size, 1];

            float[,] outputs0 = new float[1, 2];

            frameOutput = new int[raw * col];


            for (int i = 0; i < raw * col; i++)
            {
                int gr = i / col;
                int gc = i % col;

                for (int r = 0; r < image_size; r++)
                {
                    for (int c = 0; c < image_size; c++)
                    {
                        image_input[0, r, c, 0] = (float)myGrayImage[(gr * image_size) + r, (gc * image_size) + c] / 255;
                    }
                }
                interpreter.SetInputTensorData(0, image_input);
                interpreter.Invoke();
                outputs0 = new float[1, 2];
                interpreter.GetOutputTensorData(0, outputs0);
                //Debug.Log(resultArgMax(outputs0));
                //Debug.Log(outputs0[0, 0]);
                //Debug.Log(outputs0[0, 1]);
                frameOutput[i] = resultArgMax(outputs0);
            }

            // byte[] imageData = Utils.DecodeTexture(inputTex, inputTex.width, inputTex.height, 0, Flip.VERTICAL);
            // Shape shape = new Shape(1, inputTex.width, inputTex.height, 3);
            // NDArray image = new NDArray(imageData, shape);

            /*
            int input_width = inputTex.width;
            int input_height = inputTex.height;
            int raw = input_height / image_size; // int
            int col = input_width / image_size; // int

            //Debug.Log(interpreter.GetInputTensorInfo(0).shape[3]);
            //Debug.Log(input_width);1024
            //Debug.Log(input_height);499
            //Debug.Log(raw);4
            //Debug.Log(col);8


            float[,,] float_image = new float[input_height, input_width, 3];

            ToTensor(inputTex, float_image);

            float[,] grayImage = new float[input_height, input_width];

            for (int r = 0; r < input_height; r++) {

                for (int c = 0; c < input_width; c++) {

                    grayImage[r, c] = RGBToGray(float_image[r, c, 0], float_image[r, c, 1], float_image[r, c, 2]);

                }
            }

            float[,,,] image_input = new float[raw * col, image_size, image_size,1];

            int[] outputs0 = new int[raw * col];

            for (int r = 0; r < raw*image_size; r++)
            {
                for (int c = 0; c < col*image_size; c++)
                {
                    int image_r = r / image_size;
                    int inner_r = r % image_size;
                    int image_c = c / image_size;
                    int inner_c = c % image_size;

                    //Debug.Log(image_r * col + image_c);
                    //Debug.Log(inner_r);
                    //Debug.Log(inner_c);
                    image_input[image_r * col + image_c, inner_r, inner_c,0] = (float)grayImage[r, c]/255;
                }
            }

            // ToTensor(inputTex, ref input);
            Debug.Log(string.Join("a", grayImage));
            Debug.Log(image_input[14,0,0,0]);

            interpreter.SetInputTensorData(0, image_input);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            Debug.Log(outputs0);
            */
        }


        public float RGBToGray(float r, float g, float b)
        {
            return (float)(0.30 *r + 0.59*g + 0.11*b);
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

        private Texture2D TextureToTexture2D(Texture texture)
        {
            Texture2D texture2D = new Texture2D(texture.width, texture.height, TextureFormat.RGBA32, false);
            RenderTexture currentRT = RenderTexture.active;
            RenderTexture renderTexture = RenderTexture.GetTemporary(texture.width, texture.height, 32);
            Graphics.Blit(texture, renderTexture);

            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture2D.Apply();

            RenderTexture.active = currentRT;
            RenderTexture.ReleaseTemporary(renderTexture);
            return texture2D;
        }

        Texture2D RTtoTexture2D(RenderTexture rTex)
        {
            Texture2D tex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGB24, false);
            // ReadPixels looks at the active RenderTexture.
            RenderTexture.active = rTex;
            tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
            tex.Apply();
            return tex;
        }

        public int[] GetResults()
        {
            return frameOutput;
        }
    }
}
