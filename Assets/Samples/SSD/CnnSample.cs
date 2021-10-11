using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(WebCamInput))]
public class CnnSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")] string fileName = "cnn_crack_lite.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField] Text framePrefab = null;
    //[SerializeField, Range(0f, 1f)] float scoreThreshold = 0.5f;
    //[SerializeField] TextAsset labelMap = null;

    CNN cnn;
    Text[] frames;
    //string[] labels;

    void Start()
    {
        cnn = new CNN(fileName);

        // Init frames
        
        frames = new Text[200];
        var parent = cameraView.transform;
        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = Instantiate(framePrefab, Vector3.zero, Quaternion.identity, parent);
            frames[i].transform.localPosition = Vector3.zero;

            frames[i].gameObject.SetActive(false);
        }

        // Labels
        //labels = labelMap.text.Split('\n');

        GetComponent<WebCamInput>().OnTextureUpdate.AddListener(Invoke);
    }

    void OnDestroy()
    {
        GetComponent<WebCamInput>().OnTextureUpdate.RemoveListener(Invoke);
        cnn?.Dispose();
    }

    void Invoke(Texture texture)
    {
        cnn.Invoke(texture);
        var results = cnn.GetResults();
        //Debug.Log(results);

        var size = cameraView.rectTransform.rect.size;
        /*
        for (int i = 0; i < results.Length; i++)
        {
            SetFrame(frames[i], results[i], size);
        }
        */

        /*
        frames[0].gameObject.SetActive(true);
        frames[0].text = "test";
        var rt = frames[0].transform as RectTransform;
        rt.anchoredPosition = new Vector2(-0.4f, -0.4f) * size;
        rt.sizeDelta = new Vector2(120f, 120f);
        */
        setFrames(frames, 120, results);

        cameraView.material = cnn.transformMat;
    }

    void setFrames(Text[] frames, int unit_size, int[] results)
    {

        int height = cameraView.mainTexture.height;
        int width = cameraView.mainTexture.width;
        // int frame_num = (height / unit_size) * (width / unit_size);

        int raw = height / unit_size; // int
        int col = width / unit_size; // int
        for (int i = 0; i< results.Length; i++)
        {
            //float y = -((float)(i / col) / (float)raw) + 0.5f;
            //float x = ((float)(i % col) / (float)col) - 0.5f;

            //float y = ((float)(i % raw) / (float)raw) - 0.5f + ((float)unit_size / (float)height);
            //float x = ((float)(i / raw) / (float)col) - 0.5f;

            float y = ((float)(i / col) / (float)raw) - 0.5f + ((float)unit_size / (float)height);
            float x = ((float)(i % col) / (float)col) - 0.5f;

            var rt = frames[i].transform as RectTransform;
            rt.anchoredPosition = new Vector2(x, y) * cameraView.rectTransform.rect.size;
            rt.sizeDelta = new Vector2(unit_size, unit_size);
            //frames[i].text = $"{i+1} : {(int)(results[i])}%";
            frames[i].text = $"{(int)(results[i])}%";

            if (results[i] == 0)
            {
                frames[i].gameObject.SetActive(false);
            }
            else
            {
                frames[i].gameObject.SetActive(true);
            }
        }  
    }
}
