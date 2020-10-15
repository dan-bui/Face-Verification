using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using System.IO;
using System.Diagnostics;
using System.Threading;
using System.Data.SqlTypes;

namespace WindowsFormsApp3
{
    public partial class Form1 : Form
    {
        #region Variables
        private Capture videoCapture = null;
        private Image<Bgr, Byte> currentFrame = null;
        Mat frame = new Mat();
        private bool faceDetec = false;

        CascadeClassifier faceCass = new CascadeClassifier("haarcascade_frontalface_alt.xml");
        Image<Bgr, Byte> faceResult = null;
        List<Image<Gray, Byte>> Trained = new List<Image<Gray, Byte>>();
        List<int> PersonLabel = new List<int>();
        bool EnableSaveimage = false;
        
        private static bool isTrained = false;
        
        EigenFaceRecognizer recognizer;
        List<string> PersonsNames = new List<string>();
        #endregion
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            videoCapture = new Capture();
            videoCapture.ImageGrabbed += dangbui;
            videoCapture.Start();
        }
        private void dangbui(object sender, EventArgs e)
        {
            videoCapture.Retrieve(frame, 0);
            currentFrame = frame.ToImage<Bgr, Byte>().Resize(pictureBox1.Width, pictureBox1.Height, Inter.Cubic);
            if (faceDetec)
            {
                Mat grayImage = new Mat();
                CvInvoke.CvtColor(currentFrame, grayImage, ColorConversion.Bgr2Gray);
                CvInvoke.EqualizeHist(grayImage, grayImage);
                Rectangle[] faces = faceCass.DetectMultiScale(grayImage, 1.1, 3, Size.Empty, Size.Empty);
                if (faces.Length > 0)
                {
                    int faceID = 0;
                    foreach (var face in faces)
                    {
                        CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                        Image<Bgr, Byte> resultImage = currentFrame.Convert<Bgr, Byte>();
                        resultImage.ROI = face;
                        pictureBox2.SizeMode = PictureBoxSizeMode.StretchImage;
                        pictureBox2.Image = resultImage.Bitmap;
                        if(EnableSaveimage)
                        {
                            string path = Directory.GetCurrentDirectory() + @"\TrainedImages";
                            if (!Directory.Exists(path))
                                Directory.CreateDirectory(path);
                            Task.Factory.StartNew(() =>
                            {


                                for (int i = 0; i < 10; i++)
                                {
                                    resultImage.Resize(200, 200, Inter.Cubic).Save(path + @"\" + textBox1.Text + DateTime.Now.ToString("dd-mm-yyyy-hh-mm-ss") + ".jpg");
                                    Thread.Sleep(1000);
                                }
                            });
                        }
                        EnableSaveimage = false;
                        if(button2.InvokeRequired)
                        {
                            button2.Invoke(new ThreadStart(delegate {
                                button2.Enabled = true;
                            }));
                        }
                        if(isTrained)
                        {
                            Image<Gray, Byte> grayFace = resultImage.Convert<Gray, Byte>().Resize(200.200, Inter.Cubic);
                            CvInvoke.EqualizeHist(grayFace, grayFace);

                            var result = recognizer.Predict(grayFace);
                            pictureBox3.Image = grayFace.Bitmap;
                            pictureBox4.Image = Trained[result.Label].Bitmap;
                            Debug.WriteLine(result.Label + "." + result.Distance);
                            if(result.Label != -1 && result.Distance < 2000)
                            {
                                CvInvoke.PutText(currentFrame, PersonsNames[result.Label], new Point(face.X - 2, face.Y - 2),
                                    FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Green).MCvScalar, 2);
                            }
                            else
                            {
                                CvInvoke.PutText(currentFrame, "Unknow", new Point(face.X - 2, face.Y - 2),
                                    FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                            }

                        }
                    }
                }
            }
            pictureBox1.Image = currentFrame.Bitmap;
            faceDetec = true;

        }

        private void button2_Click(object sender, EventArgs e)
        {
            button3.Enabled = true;
            button2.Enabled = false;
            EnableSaveimage = true ;
            if (pictureBox2.Image != null) 
            {
                MessageBox.Show("OK");
            }
            else
            {
                MessageBox.Show("fuck");
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            button3.Enabled = false;
            button2.Enabled = true;
            EnableSaveimage = false;
        }

        private void button4_Click(object sender, EventArgs e)
        {
            TrainImage();
        }
        private bool TrainImage()
        {
            int imageCount = 0;
            double Threshold = -1;
            Trained.Clear();
            PersonLabel.Clear();
            try
            {
                string path = Directory.GetCurrentDirectory() + @"\TrainedFaces";
                string[] files = Directory.GetFiles(path, "*.jpg", SearchOption.AllDirectories);
                foreach(var file in files)
                {
                    Image<Gray, Byte> trainedImage = new Image<Gray, byte>(file);
                    Trained.Add(trainedImage);
                    PersonLabel.Add(imageCount);
                    PersonLabel.Add(imageCount);
                    
                    imageCount++;
                }
                EigenFaceRecognizer recognizer = new EigenFaceRecognizer(imageCount, Threshold);
                recognizer.Train(Trained.ToArray(), PersonLabel.ToArray());

                isTrained = true;
                Debug.WriteLine(imageCount);
                Debug.WriteLine(isTrained);
                return isTrained = true;
            }
            catch(Exception ex)
            {
                isTrained = false;
                MessageBox.Show("Erro" + ex.Message);
                return false;
            }
        }
    }
}