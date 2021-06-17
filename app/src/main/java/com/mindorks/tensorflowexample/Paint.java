

package com.mindorks.tensorflowexample;


import android.graphics.PointF;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.mindorks.tensorflowexample.view.DrawModel;
import com.mindorks.tensorflowexample.view.DrawView;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class Paint extends AppCompatActivity implements View.OnTouchListener, View.OnClickListener {
    TextView textView;
    List<String> pictures;
    private TextView textView2;
    private static final String TAG = "Paint";
    Boolean a = true;

    private static final int PIXEL_WIDTH = 28;
    private static final int PIXEL_HEIGHT = 28;

    private TextView mResultText;

    private float mLastX;

    private float mLastY;

    private DrawModel mModel;
    private DrawView mDrawView;




    private PointF mTmpPoint = new PointF();

    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "file:///android_asset/mnist_model_graph.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/graph_label_strings.txt";

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();


    @SuppressWarnings("SuspiciousNameCombination")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mModel = new DrawModel(PIXEL_HEIGHT, PIXEL_WIDTH);

        mDrawView = (DrawView) findViewById(R.id.view_draw);
        mDrawView.setModel(mModel);
        mDrawView.setOnTouchListener(this);



        View clearButton = findViewById(R.id.buttonClear);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClearClicked();
            }
        });

        mResultText = (TextView) findViewById(R.id.textResult);

        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    makeButtonVisible();
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {

            }
        });
    }

    @Override
    protected void onResume() {
        mDrawView.onResume();
        super.onResume();
    }

    @Override
    protected void onPause() {
        mDrawView.onPause();
        super.onPause();
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction() & MotionEvent.ACTION_MASK;

        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;

        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;

        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();
        mDrawView.calcPos(mLastX, mLastY, mTmpPoint);
        float lastConvX = mTmpPoint.x;
        float lastConvY = mTmpPoint.y;
        mModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        mDrawView.calcPos(x, y, mTmpPoint);
        float newConvX = mTmpPoint.x;
        float newConvY = mTmpPoint.y;
        mModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        mDrawView.invalidate();
    }

    private void processTouchUp() {
        mModel.endLine();
    }

    private void onDetectClicked() {
        float pixels[] = mDrawView.getPixelData();

        final List<Classifier.Recognition> results = classifier.recognizeImage(pixels);

        if (results.size() > 0) {
            String value = " Number is : " +results.get(0).getTitle();
            mResultText.setText(value);
        }

    }

    private void onClearClicked() {
        mModel.clear();
        mDrawView.reset();
        mDrawView.invalidate();

        mResultText.setText("");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }
    public void onClick(final View v) {
        textView2 = (TextView) findViewById(R.id.textView2);
        new CountDownTimer(10000, 1000) {
            public void onTick(long millisUntilFinished) {
                textView2.setText("         "
                        + millisUntilFinished / 1000+"       ");
                float pixels[] = mDrawView.getPixelData();

                final List<Classifier.Recognition> results = classifier.recognizeImage(pixels);

              if (results.size() > 0) {
                  String value = results.get(0).getTitle();
                  mResultText.setText(value);




              }
            }
            public void onFinish() {
                textView2.setText("");
                mModel.clear();
                mDrawView.reset();
                mDrawView.invalidate();

                mResultText.setText("");
            }
        }
                .start();
        Button btn = (Button)   findViewById(R.id.button);
        v.setVisibility(View.INVISIBLE);
        v.postDelayed(new Runnable() {
            @Override
            public void run() {
                v.setVisibility(View.VISIBLE);
            }
        }, 10 * 1000);
        textView = (TextView) findViewById(R.id.textVew);
        pictures = new ArrayList<>();
        pictures.add("1");
        pictures.add("2");
        pictures.add("3");
        pictures.add("4");
        pictures.add("5");
        pictures.add("6");
        pictures.add("7");
        pictures.add("8");
        pictures.add("9");
        pictures.add("10");


        Collections.shuffle(pictures);
        textView.setVisibility(View.VISIBLE);
        textView.postDelayed(new Runnable() {
            @Override
            public void run() {
                textView.setVisibility(View.GONE);
            }
        }, 10 * 1000);
        if (a) {
            String current = pictures.get(0);
            if (current.equals("1")) {
                textView.setText("0");
            }  if (current.equals("2")) {
                textView.setText("1");
            }  if (current.equals("3")) {
                textView.setText("2");
            }  if (current.equals("4")) {
                textView.setText("3");
            }  if (current.equals("5")) {
                textView.setText("4");
            }  if (current.equals("6")) {
                textView.setText("5");
            }  if (current.equals("7")) {
                textView.setText("6");
            }  if (current.equals("8")) {
                textView.setText("7");
            }  if (current.equals("9")) {
                textView.setText("8");  }  if (current.equals("10")) {
                textView.setText("9");


            }
            }

            }


                //спользуется рандомное слово в array list, но по нажатию на text view. ужно сделать, что бы каждые 10 сек слово менялось
            }









