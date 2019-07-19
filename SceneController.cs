using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneController : MonoBehaviour
{
    public ImageSynthesis synth;
    public GameObject[] prefabs;
    public int num_Objects = 8;

    public int trainingImages;
    public int valImages;

    public bool grayscale = false;
    public bool save = false;

    private ShapePool pool;

    private int frameCount = 0;

    // Start is called before the first frame update
    void Start()
    {
        pool = ShapePool.Create(prefabs);
    }

    // Update is called once per frame
    void Update()
    {
        if (frameCount < trainingImages + valImages)
        {
            /*
                if (frameCount % 5 == 0) {

                GenerateRandom();
                Debug.Log($"FrameCount: {frameCount}");

                }
            */
            GenerateRandom();
            Debug.Log($"FrameCount: {frameCount}");
            frameCount++;

            if (save)
            {
                if (frameCount < trainingImages)
                {
                    string filename = $"image_{frameCount.ToString().PadLeft(5, '0')}";
                    synth.Save(filename, 1920, 1080, "captures/train", 2);
                }
                else if (frameCount < trainingImages + valImages)
                {
                    int valFrameCount = frameCount - trainingImages;
                    string filename = $"image_{valFrameCount.ToString().PadLeft(5, '0')}";
                    synth.Save(filename, 1920, 1080, "captures/val", 2);
                }

            }


        }
    }

    void GenerateRandom()
    {
        pool.ReclaimAll();
        int objectsThisTime = 8;
        for (int i = 0; i < objectsThisTime; i++)
        {
            //pick out prefab
            int prefabIndx = 0;
            if (i == 0 || i == 1)
            {
                prefabIndx = 0;
            }
            else if (i == 2 || i == 3)
            {
                prefabIndx = 1;
            }
            else if (i == 4 || i == 6) //Careful to include the correct index here, in the Scene Controller prefab2 should be Jawlft and prefab3 should be jaw right
            {
                prefabIndx = 2;
            }
            else
            {
                prefabIndx = 3;
            }

            GameObject prefab = prefabs[prefabIndx];


            if (i == 0)
            {
                //OBJECT 1 SHAFT RIGHT ARM
                /*
                DATA from Python script
                JOINT 4 PSM_1 SHAFT RIGHT ARM Orientation in quaternions  WXYZ [ 0.06844   0.458484 -0.60438   0.647945]
                JOINT 4 PSM_1 SHAFT RIGHT ARM Position in mm [14.620537  3.197187 64.227186]
                
                */

                //Position
                float newX, newY, newZ;
                newX = 14.620537f;
                newY = 3.197186f;
                newZ = 64.227186f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = 0.458484f;
                float quatY = -0.60438f;
                float quatZ = 0.647945f;
                float quatW = 0.06844f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */
            }
            else if (i == 1)
            {

                //OBJECT 2 SHAFT LEFT ARM
                /*
                DATA from Python script
               
                JOINT 4 PSM_2 SHAFT LEFT ARM Orientation in quaternions  WXYZ [ 0.647431 -0.618062 -0.015413 -0.44564 ]
                JOINT 4 PSM_2 SHAFT LEFT ARM Position in mm [-29.382017  11.433768  96.028443]
                
                */

                //Position
                float newX, newY, newZ;
                newX = -29.382016f;
                newY = 11.433768f;
                newZ = 96.028442f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = -0.618062f;
                float quatY = -0.015413f;
                float quatZ = -0.44564f;
                float quatW = 0.647431f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */

            }
            else if (i == 2)
            {

                //OBJECT 3 LOGO BODY RIGHT ARM
                /*
                DATA from Python script
                
                JOINT 5 PSM_1 LOGO BODY RIGHT ARM Orientation in quaternions  WXYZ [ 0.85215  -0.171896  0.181329 -0.459795]
                JOINT 5 PSM_1 LOGO BODY RIGHT ARM Position in mm [14.620537  3.197187 64.227186]
                
                */

                //Position
                float newX, newY, newZ;
                newX = 14.620537f;
                newY = 3.197187f;
                newZ = 64.227186f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = -0.171896f;
                float quatY = 0.181329f;
                float quatZ = -0.459795f;
                float quatW = 0.85215f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */

            }
            else if (i == 3)
            {

                //OBJECT 4 LOGO BODY LEFT ARM
                /*
                DATA from Python script
                
                JOINT 5 PSM_2 LOGO BODY LEFT ARM Orientation in quaternions  WXYZ [ 0.873159  0.055205 -0.104977  0.472785]
                JOINT 5 PSM_2 LOGO BODY RIGHT ARM Position in mm [-29.382017  11.433768  96.028443]
                */

                //Position
                float newX, newY, newZ;
                newX = -29.382017f;
                newY = 11.433768f;
                newZ = 96.028443f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = 0.055205f;
                float quatY = -0.104977f;
                float quatZ = 0.472785f;
                float quatW = 0.873159f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */

            }
            else if (i == 4)
            {

                //OBJECT 5  LEFT-JAW RIGHT-ARM 

                /*
                DATA from Python script
                
                OLD - JAWS_TOGETHER
                JOINT 6 PSM_1 JAW RIGHT ARM Orientation in quaternions  WXYZ [ 0.207232  0.472756  0.228358 -0.825475]
                JOINT 6 PSM_1 JAW RIGHT ARM Position in mm [ 8.056809 -1.517353 68.410556]
                
                NEW
                JOINT 6 PSM_1 LEFT-JAW RIGHT-ARM Orientation in quaternions  WXYZ [ 0.096489  0.507083  0.038358 -0.85562 ]
                JOINT 6 PSM_1 LEFT-JAW RIGHT-ARM Position in mm [ 8.056809 -1.517353 68.410556]
               
                

                */

                //Position
                float newX, newY, newZ;
                newX = 8.056809f;
                newY = -1.517353f;
                newZ = 68.410556f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = 0.507083f;
                float quatY = 0.038358f;
                float quatZ = -0.85562f;
                float quatW = 0.096489f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */

            }
            else if (i == 5)
            {

                //OBJECT 6 RIGHT-JAW RIGHT-ARM
                /*
                DATA from Python script
                OLD - JAWS_TOGETHER
                JOINT 6 PSM_1 JAW RIGHT ARM Orientation in quaternions  WXYZ [ 0.207232  0.472756  0.228358 -0.825475]
                JOINT 6 PSM_1 JAW RIGHT ARM Position in mm [ 8.056809 -1.517353 68.410556]

                NEW
                JOINT 6 PSM_1 RIGHT-JAW RIGHT-ARM Orientation in quaternions  WXYZ [ 0.30752   0.414578  0.406837 -0.753684]
                JOINT 6 PSM_1 RIGHT-JAW RIGHT-ARM Position in mm [ 8.056809 -1.517353 68.410556]
                
                */

                //Position
                float newX, newY, newZ;
                newX = 8.056809f;
                newY = -1.517353f;
                newZ = 68.410556f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = 0.414578f;
                float quatY = 0.406837f;
                float quatZ = -0.753684f;
                float quatW = 0.30752f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */

            }
            else if (i == 6)
            {

                //OBJECT 7 LEFT-JAW LEFT-ARM
                /*
                DATA from Python script
                OLD - JAWS_TOGETHER
                JOINT 6 PSM_2 JAW LEFT ARM Orientation in quaternions  WXYZ [ 0.684541  0.311392  0.642027 -0.149132]
                JOINT 6 PSM_2 JAW LEFT ARM Position in mm [-21.763284   6.4574    96.054437]

                NEW
                JOINT 6 PSM_2 LEFT-JAW LEFT-ARM Orientation in quaternions  WXYZ [ 0.581031  0.477455  0.581818 -0.309721]
                JOINT 6 PSM_2 LEFT-JAW LEFT-ARM Position in mm [-21.763284   6.4574    96.054437]

                */

                //Position
                float newX, newY, newZ;
                newX = -21.763284f;
                newY = 6.4574f;
                newZ = 96.054437f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = 0.477455f;
                float quatY = 0.581818f;
                float quatZ = -0.309721f;
                float quatW = 0.581031f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */

            }
            else
            {

                //OBJECT 8 RIGHT-JAW LEFT-ARM
                /*
                DATA from Python script
                OLD - JAWS_TOGETHER
                JOINT 6 PSM_2 JAW LEFT ARM Orientation in quaternions  WXYZ [ 0.684541  0.311392  0.642027 -0.149132]
                JOINT 6 PSM_2 JAW LEFT ARM Position in mm [-21.763284   6.4574    96.054437]

                NEW_DATA
                JOINT 6 PSM_2 RIGHT-JAW LEFT-ARM Orientation in quaternions  WXYZ [0.741704 0.124245 0.658767 0.021553]
                JOINT 6 PSM_2 RIGHT-JAW LEFT-ARM Position in mm [-21.763284   6.4574    96.054437]

                */

                //Position
                float newX, newY, newZ;
                newX = -21.763284f;
                newY = 6.4574f;
                newZ = 96.054437f;
                var newPos = new Vector3(newX, newY, newZ);

                var shape = pool.Get((ShapeLabel)prefabIndx);
                var newObj = shape.obj;
                newObj.transform.position = newPos;

                //Rotation

                float quatX = 0.124245f;
                float quatY = 0.658767f;
                float quatZ = 0.021553f;
                float quatW = 0.741704f;
                newObj.transform.rotation = new Quaternion(quatX, quatY, quatZ, quatW);

                //Scale
                float sx = 100.0f;
                Vector3 newScale = new Vector3(sx, sx, sx);
                newObj.transform.localScale = newScale;

                // Color
                /*
                float newR, newG, newB;
                newR = Random.Range(0.0f, 1.0f);
                newG = Random.Range(0.0f, 1.0f);
                newB = Random.Range(0.0f, 1.0f);

                var newColor = new Color(newR, newG, newB);
                newObj.GetComponent<Renderer>().material.color = newColor;
                */

            }
        }
        synth.OnSceneChange(grayscale);
    }
}
