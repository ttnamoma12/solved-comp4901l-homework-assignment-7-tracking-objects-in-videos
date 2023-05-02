Download Link: https://assignmentchef.com/product/solved-comp4901l-homework-assignment-7-tracking-objects-in-videos
<br>
<h1>Overview</h1>

One incredibly important aspect of human and animal vision is the ability to follow objects and people in our view. Whether it is a tiger chasing its prey, or you trying to catch a basketball, tracking is so integral to our everyday lives that we forget how much we rely on it. In this assignment, you will be implementing an algorithm that will track an object in a video.

You will first implement the Lucas-Kanade tracker, and then a more computationally efficient version called the Matthew-Baker (or inverse compositional) method. This method is one of the most commonly used methods in computer vision due to its simplicity and wide applicability. We have provided two video sequences: a car on a road, and a helicopter approaching a runway.

To initialize the tracker you need to define a template by drawing a bounding box around the object to be tracked in the first frame of the video. For each of the subsequent frames the tracker will update an affine transform that warps the current frame so that the template in the first frame is aligned with the warped current frame.

For extra credit, we will also look at ways to make tracking more robust, by incorporating illumination invariance and implementing the algorithm in a pyramid fashion.

<h1>Preliminaries</h1>

An image transformation or warp is an operation that acts on pixel coordinates and maps pixel values from one place to another in an image. Translation, rotation and scaling are all examples of warps. We will use the symbol <strong>W </strong>to denote warps. A warp function <strong>W </strong>has a set of parameters <strong>p </strong>associated with it and maps a pixel with coordinates <strong>x </strong>= [<em>u v</em>]<em><sup>T </sup></em>to <strong>x</strong><sup>0 </sup>= [<em>u</em><sup>0 </sup><em>v</em><sup>0</sup>]<em><sup>T</sup></em>.

<strong>x</strong><sup>0 </sup>= <strong>W</strong>(<strong>x</strong>;<strong>p</strong>)                                                                                     (1)

An affine transform is a warp that can include any combination of translation, anisotropic scaling and rotations. An affine warp can be parametrized in terms of 6 parameters <strong>p </strong>= [<em>p</em><sub>1 </sub><em>p</em><sub>2 </sub><em>p</em><sub>3 </sub><em>p</em><sub>4 </sub><em>p</em><sub>5 </sub><em>p</em><sub>6</sub>]<em><sup>T</sup></em>.

One of the convenient things about an affine transformation is that it is linear; its action on a point with coordinates <strong>x </strong>= [<em>u v</em>]<em><sup>T </sup></em>can be described as a matrix operation

 <em>u</em>0                        <em>u </em>

 <em>v</em>0  = <strong>W</strong>(<strong>p</strong>) <em>v </em>                                                                              (2)

                                   

1                               1

where <strong>W</strong>(<strong>p</strong>) is a 3 × 3 matrix such that

                                            

1 + <em>p</em>1              <em>p</em>3             <em>p</em>5

<strong>W</strong>(<strong>p</strong>) = <sup>        </sup><em>p</em><sub>2              </sub>1 + <em>p</em><sub>4         </sub><em>p</em><sub>6 </sub><sup>                                                                </sup>(3)

                                            

0               0           1

Note that for convenience when we want to refer to the warp as a function we will use <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) and when we want to refer to the matrix for an affine warp we will use <strong>W</strong>(<strong>p</strong>). Table 1 contains a summary of the variables used in the next two sections. It will be useful to keep these in mind.

Table 1: Summary of Variables

<table width="488">

 <tbody>

  <tr>

   <td width="63">Symbol</td>

   <td width="138">Vector/Matrix Size</td>

   <td width="287">Description</td>

  </tr>

  <tr>

   <td width="63"><em>u</em></td>

   <td width="138">1 × 1</td>

   <td width="287">Image horizontal coordinate</td>

  </tr>

  <tr>

   <td width="63"><em>v</em></td>

   <td width="138">1 × 1</td>

   <td width="287">Image vertical coordinate</td>

  </tr>

  <tr>

   <td width="63"><strong>x</strong></td>

   <td width="138">2 × 1 or 1 × 1</td>

   <td width="287">pixel coordinates: (<em>u,v</em>) or unrolled</td>

  </tr>

  <tr>

   <td width="63"><strong>I</strong></td>

   <td width="138"><em>m </em>× 1</td>

   <td width="287">Image unrolled into a vector (<em>m </em>pixels)</td>

  </tr>

  <tr>

   <td width="63"><strong>T</strong></td>

   <td width="138"><em>m </em>× 1</td>

   <td width="287">Template unrolled into a vector (<em>m </em>pixels)</td>

  </tr>

  <tr>

   <td width="63"><strong>W</strong>(<strong>p</strong>)</td>

   <td width="138">3 × 3</td>

   <td width="287">Affine warp matrix</td>

  </tr>

  <tr>

   <td width="63"> </td>

   <td width="138">6 × 1</td>

   <td width="287">parameters of affine warp</td>

  </tr>

  <tr>

   <td width="63"></td>

   <td width="138"><em>m </em>× 1</td>

   <td width="287">partial derivative of image wrt <em>u</em></td>

  </tr>

  <tr>

   <td width="63"><em>∂v</em></td>

   <td width="138"><em>m </em>× 1</td>

   <td width="287">partial derivative of image wrt <em>v</em></td>

  </tr>

  <tr>

   <td width="63"><em><u>∂</u></em><strong><u>T </u></strong><em>∂u</em></td>

   <td width="138"><em>m </em>× 1</td>

   <td width="287">partial derivative of image wrt <em>u</em></td>

  </tr>

  <tr>

   <td width="63"><em><u>∂</u></em><strong><u>T</u></strong><em>∂v</em></td>

   <td width="138"><em>m </em>× 1</td>

   <td width="287">partial derivative of image wrt <em>u</em></td>

  </tr>

  <tr>

   <td width="63">∇<strong>I</strong></td>

   <td width="138"><em>m </em>× 2</td>

   <td width="287">image gradient ∇<strong>I</strong>(<strong>x</strong>) = <em><sub>∂u</sub><u><sup>∂</sup></u></em><strong><u><sup>I               </sup></u></strong><em><sub>∂v</sub><u><sup>∂</sup></u></em><strong><u><sup>I</sup></u></strong><sup>i </sup>h</td>

  </tr>

  <tr>

   <td width="63">∇<strong>T</strong></td>

   <td width="138"><em>m </em>× 2</td>

   <td width="287">image gradient</td>

  </tr>

  <tr>

   <td width="63"><em><u>∂</u></em><strong><u>W</u></strong><em>∂</em><strong>p</strong></td>

   <td width="138">2 × 6</td>

   <td width="287">Jocobian of affine warp wrt its parameters</td>

  </tr>

  <tr>

   <td width="63"><strong>J</strong></td>

   <td width="138"><em>m </em>× 6</td>

   <td width="287">Jacobian of error function <em>L </em>wrt <strong>p</strong></td>

  </tr>

  <tr>

   <td width="63"><strong>H</strong></td>

   <td width="138">6 × 6</td>

   <td width="287">Pseudo Hessian of <em>L </em>wrt <strong>p</strong></td>

  </tr>

 </tbody>

</table>

<h1>Lucas-Kanade: Forward Additive Alignment</h1>

A Lucas Kanade tracker maintains a warp <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) which aligns a sequence of images <strong>I</strong><em><sub>t </sub></em>to a template <strong>T</strong>. We denote pixel locations by <strong>x</strong>, so <strong>I</strong>(<strong>x</strong>) is the pixel value at location <strong>x </strong>in image <strong>I</strong>. For the purposes of this derivation, <strong>I </strong>and <strong>T </strong>are treated as column vectors (think of them as unrolled image matrices). <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) is the point obtained by warping <strong>x </strong>with a transform that has parameters <strong>p</strong>. <strong>W </strong>can be any transformation that is continuous in its parameters <strong>p</strong>. Examples of valid warp classes for <strong>W </strong>include translations (2 parameters), affine transforms (6 parameters) and full projective transforms (8 parameters). The Lucas Kanade tracker minimizes the pixel-wise sum of square difference between the warped image <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)) and the template <strong>T</strong>.

In order to align an image or patch to a reference template, we seek to find the parameter vector <strong>p </strong>that minimizes <em>L</em>, where:

<em>L </em>= <sup>X</sup>[<strong>T</strong>(<em>x</em>) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>))]<sup>2                                                                                                             </sup>(4)

<strong>x</strong>

In general this is a difficult non-linear optimization, but if we assume we already have a close estimate <strong>p </strong>of the correct warp, then we can assume that a small linear change ∆<em>p </em>is enough to get the best alignment. This is the forward additive form of the warp. The objective can then be written as:

<table width="601">

 <tbody>

  <tr>

   <td width="583"><em>L </em>≈ <sup>X</sup>[<strong>T</strong>(<em>x</em>) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p </strong>+ ∆<strong>p</strong>))]<sup>2</sup><strong>x</strong>Expanding this to the first order with Taylor Series gives us:</td>

   <td width="19">(5)</td>

  </tr>

  <tr>

   <td width="583"><em>∂</em><strong>W</strong><em>L </em>≈ <sup>X</sup>[<strong>T</strong>(<em>x</em>) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)) − ∇<strong>I</strong>(<strong>x</strong>)        ∆<strong>p</strong>]<sup>2</sup></td>

   <td width="19">(6)</td>

  </tr>

 </tbody>

</table>

<strong>x                                                                                                 </strong><em>∂</em><strong>p</strong>

Here, which is the vector containing the horizontal and vertical gradient at pixel location <strong>x</strong>. Rearranging the Taylor expansion, it can be rewritten as a typical least squares approximation ∆<strong>p</strong><sup>∗ </sup>= argmin<sub>∆<strong>p </strong></sub>||<em>A</em>∆<em>p </em>− <em>b</em>||<sup>2</sup>

∆<strong>p</strong><sup>∗ </sup>= argmin<sup>X</sup>(7)

∆<strong>p</strong>

<strong>x</strong>

This can be solved with ∆<em>p</em><sup>∗ </sup>= (<em>A<sup>T</sup>A</em>)<sup>−1</sup><em>A<sup>T</sup>b </em>where:

(<em>A<sup>T</sup>A</em>) = <strong>H </strong>= <sup>X</sup>                                                              (8)

<strong>x </strong>   (9)

<strong>x</strong>

<em>b         </em>= <strong>T</strong>(<strong>x</strong>) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>))                                                                   (10)

Once ∆<strong>p </strong>is computed, the best estimate warp can be updated <strong>p </strong>← <strong>p </strong>+ ∆<strong>p</strong>, and the whole procedure can be repeated again, stopping when ∆<em>p </em>is less than some threshold.

<h1>Matthew-Baker: Inverse Compositional Alignment</h1>

While Lucas-Kanade alignment works very well, it is computationally expensive. The inverse compositional method is similar, but requires less computation, as the Hessian and Jacobian only need to be computed once. One caveat is that the warp needs to be invertible. Since affine warps are invertible, we can use this method.

In the previous section, we combined two warps by simply adding one parameter vector to another parameter vector, and produce a new warp <strong>W</strong>(<strong>x</strong><em>,</em><strong>p </strong>+ <strong>p</strong><sup>0</sup>). Another way of combining warps is through composition of warps. After applying a warp <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) to an image, another warp <strong>W</strong>(<strong>x</strong>;<strong>q</strong>) can be applied to the warped image. The resultant (combined) warp is

<strong>W</strong>(<strong>x</strong>;<strong>q</strong>) ◦ <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) = <strong>W</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)<em>,</em><strong>q</strong>)                                                            (11)

Since affine warps can be implemented as matrix multiplications, composing two affine warps reduces to multiplying their corresponding matrices

<strong>W</strong>(<strong>x</strong>;<strong>q</strong>) ◦ <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) = <strong>W</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)<em>,</em><strong>q</strong>) = <strong>W</strong>(<strong>W</strong>(<strong>p</strong>)<strong>x</strong><em>,</em><strong>q</strong>) = <strong>W</strong>(<strong>q</strong>)<strong>W</strong>(<strong>p</strong>)<strong>x                             </strong>(12)

An affine transform can also be inverted. The inverse warp of <strong>W</strong>(<strong>p</strong>) is simply the matrix inverse of <strong>W</strong>(<em>p</em>), <strong>W</strong>(<strong>p</strong>)<sup>−1 </sup>. In this assignment it will sometimes be simpler to consider an affine warp as a set of 6 parameters in a vector <strong>p </strong>and it will sometimes be easier to work with the matrix version <strong>W</strong>(<strong>p</strong>). Fortunately, switching between these two forms is easy (Equation 3).

The minimization is performed using an iterative procedure by making a small change (∆<strong>p</strong>) to <strong>p </strong>at each iteration. It is computationally more efficient to do the minimization by finding the <strong>p </strong>that helps align the template to the image, than applying the inverse warp to the image. This is because the image will change with each frame of the video, but the template is fixed at initialization. We will see soon that doing this allows us to write the Hessian and Jacobian in terms of the template, and so this can be computed once at the beginning of the tracking. Hence at each step, we want to find the <strong>p </strong>to minimize

<em>L </em>= <sup>X</sup>[<strong>T</strong>(<strong>W</strong>(<strong>x</strong>;∆<em>p</em>)) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>))]<sup>2                                                                                           </sup>(13)

<strong>x</strong>

For tracking a patch template, the summation is performed only over the pixels lying inside the template region. We can expand <strong>T</strong>(<strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)) in terms of its first order linear approximation to




get

2

(<strong>x</strong>;∆<strong>p</strong>))                                                (14)

<strong>x</strong>

h                          i




where <strong>T</strong>(<em>x</em>) = <em><sup>∂</sup></em><strong><sup>T</sup></strong><em><sub>∂u</sub></em><sup>(<strong>x</strong>) <em>∂</em><strong>T</strong></sup><em><sub>∂v</sub></em><sup>(<strong>x</strong>) </sup>. To minimize we need to take the derivative of <em>L </em>and set it to zero

<em>∂L</em>

(15)

<strong>x</strong>

Setting to zero, switching from summation to vector notation and solving for ∆<em>p </em>we get

∆<strong>p </strong>= <strong>H</strong><sup>−1</sup><strong>J</strong><em><sup>T</sup></em>[<strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)) − <strong>T</strong>]                                                                       (16)

where <strong>J </strong>is the Jacobian of <strong>T</strong>(<strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)), <em>J </em>= <strong>T</strong><em><u><sup>∂</sup></u><sub>∂</sub></em><strong><u><sup>W</sup></u><sub>p </sub></strong>, <strong>H </strong>is the approximated Hessian <strong>H </strong>= <strong>J</strong><em><sup>T</sup></em><strong>J </strong>and <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>)) is the warped image. Note that for a given template, the Jacobian <strong>J </strong>and Hessian <strong>H </strong>are independent of <strong>p</strong>. This means they only need to be computed once and then they can be reused during the entire tracking sequence.

Once ∆<strong>p </strong>has been solved for, it needs to be inverted and composed with <strong>p </strong>to get the new warp parameters for the next iteration.

<strong>W</strong>(<strong>x</strong>;<strong>p</strong>) ← <strong>W</strong>(<strong>x</strong>;<strong>p</strong>) ◦ <strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)<sup>−1                                                                                                  </sup>(17)

The next iteration solves Equation 16 starting with the new value of <strong>p</strong>. Possible termination criteria include the absolute value of ∆<em>p </em>falling below some value or running for some fixed number of iterations.

<h1>1       Theory Questions                                                                                     (25 points)</h1>

<strong>Type down your answers for the following questions in your write-up. </strong>Each question should only take a couple of lines. In particular, the “proofs” do not require any lengthy calculations. If you are lost in many lines of complicated algebra you are doing something much too complicated (or wrong).

<h2>Q1.1: Calculating the Jacobian                                                                                   (15 points)</h2>

Assuming the affine warp model defined in Equation 3, derive the expression for the Jacobian Matrix <strong>J </strong>in terms of the warp parameters <strong>p </strong>= [<em>p</em><sub>1 </sub><em>p</em><sub>2 </sub><em>p</em><sub>3 </sub><em>p</em><sub>4 </sub><em>p</em><sub>5 </sub><em>p</em><sub>6</sub>]<sup>0 </sup>.

<h2>Q1.2: Computational complexity                                                                               (10 points)</h2>

Find the computational complexity (Big O notation) for the initialization step (Precomputing <strong>J </strong>and <strong>H</strong><sup>−1</sup>) and for each runtime iteration (Equation 16) of the Inverse Compositional method. Express your answers in terms of <em>n</em>, <em>m </em>and <em>p </em>where <em>n </em>is the number of pixels in the template <strong>T</strong>, <em>m </em>is the number of pixels in an input image <strong>I </strong>and <em>p </em>is the number of parameters used to describe the warp <strong>W</strong>. How does this compare to the run time of the regular Lucas-Kanade method?

<h1>2       Lucas-Kanade Tracker                                                                          (60 points)</h1>

For this section, TAs will grade your tracker based on the performance you achieved on the two provided video sequences: (1) data/car1/ and (2) data/landing/. The provided script files lk demo.m and mb demo.m handle reading in images, template region marking, making tracker function calls and displaying output onto the screen. The function prototypes provided are guidelines. Please make sure that your code runs functionally with the original script and generates the outputs we are looking for (a frame sequence with the bounding box of the target being tracked on each frame) so that we can replicate your results.

Note that the only thing TAs would do for you during grading is change the input data directory, and initialize your tracker based on what you mentioned in your write-up. Please submit one video for each of them in the results/ directory, with file name car.mp4 and landing.mp4. Also, please mention the initialization coordinates of your tracker for both video sequences in your write-up and in your code.

<h2>Q2.1: Write a Lucas-Kanade Tracker for a Flow Warp                                         (20 points)</h2>

Write the function with the following function signature:

[u,v] = LucasKanade(It, It1, rect) that computes the optimal local motion from frame <strong>I</strong><em><sub>t </sub></em>to frame <strong>I</strong><em><sub>t</sub></em><sub>+1 </sub>that minimizes Equation 1. Here It is the image frame It, It1 is the image frame <strong>I</strong><em><sub>t</sub></em><sub>+1 </sub>, and rect is the 4 × 1 vector that represents a rectangle on the image frame It. The four components of the rectangle are [x, y, w, h], where (x, y) is the top-left corner and (w, h) is the width and height of the bounding box. The rectangle is inclusive, i.e., in includes all the four corners. To deal with fractional movement of the template, you will need to interpolate the image using the Matlab function interp2. You will also need to iterate the estimation until the change in warp parameters (u, v) is below a threshold. Use the forward compositional (Lucas-Kanade method) for this question.

<h2>Q2.2: Initializing the Matthew-Baker Tracker                                                       (10 points)</h2>

Write the function initAffineMBTracker() that initializes the inverse compositional tracker by precomputing important matrices needed to track a template patch.

function [affineMBContext] = initAffineMBTracker(img, rect)

The function will input a greyscale image (img) along with a bounding box (rect) (in the format [x y w h]).

The function should output a Matlab structure affineMBContext that contains the Jacobian of the affine warp with respect to the 6 affine warp parameters and the inverse of the approximated Hessian matrix (<strong>J </strong>and <strong>H</strong><sup>−1 </sup>in Equation 16).

<h2>Q2.3: The Main Matthew-Baker Tracker                                                                 (20 points)</h2>

Write the function affineMBTracker() that does the actual template tracking. function [Wout] = affineMBTracker(img, tmp, rect, Win, context)

The function will input a greyscale image of the current frame (img), the template image (tmp), the bounding box rect that marks the template region in tmp, The affine warp matrix for the previous frame (Win) and the precomputed <strong>J </strong>and <strong>H</strong><sup>−1 </sup>matrices context.

The function should output the 3 × 3 matrix Wout that contains the new affine warp matrix updated so that it aligns the current frame with the template.

You can either used a fixed number of gradient descent iterations or formulate a stopping criteria for the algorithm. You can use the included image warping function to apply affine warps to images.

<h2>Q2.4: Tracking a Car                                                                                                       (5 points)</h2>

Test your trackers on the short car video sequence (data/car1/) by running the wrapper scripts lk demo.m and mb demo.m. What sort of templates work well for tracking? At what point does the tracker break down? Why does this happen?

<strong>In your write-up: </strong>Submit your best video of the car being tracked. Save it as results/car.mp4.

<h2>Q2.5: Tracking Runway Markings                                                                               (5 points)</h2>

Try running your tracker on the landing video (data/landing/). This video was taken during a runway approach. With a model of the markings (the lengths of the line segments etc) the output of the tracker can be used to estimate the camera position with respect to the runway and can be used in an automated landing system.

<strong>In your write-up: </strong>Submit your best video of the runway markings being tracked. Save it as results/landing.mp4.

Figure 2: Tracking in the car and runway image sequences

<table width="624">

 <tbody>

  <tr>

   <td width="521"><strong>3       Extra Credit</strong></td>

   <td width="103"><strong>(40 points)</strong></td>

  </tr>

  <tr>

   <td width="521"><strong>Q3.1x: Adding Illumination Robustness</strong></td>

   <td width="103"><strong>(20 points)</strong></td>

  </tr>

 </tbody>

</table>

The LK tracker as it is formulated now breaks down when there is a change in illumination because the sum of squared distances error it tries to minimize is sensitive to illumination changes. There are a couple of things you could try to do to fix this. The first is to scale the brightness of pixels in each frame so that the average brightness of pixels in the tracked region stays the same as the average brightness of pixels in the template. The second is to use a more robust M-estimator instead of least squares (so, e.g. a Huber or a Tukey M-estimator) that does not let outliers adversely affect the cost function evaluation. Note that doing this will modify our least squares problem to a weighted least squares problem, i.e. for each residual term you will have a corresponding weight <strong>Λ</strong><em><sub>ii </sub></em>and your minimization function will look like

<em>L </em>= <sup>X</sup>Λ<em><sub>ii</sub></em>[<strong>T</strong>(<strong>W</strong>(<strong>x</strong>;∆<strong>p</strong>)) − <strong>I</strong>(<strong>W</strong>(<strong>x</strong>;<strong>p</strong>))]<sup>2                                                                                              </sup>(18)

<strong>x</strong>

leading your jacobian computation to be a weighted jacobian instead of what you have seen earlier (Eq. 8)

<table width="398">

 <tbody>

  <tr>

   <td width="372"><em>A<sup>T</sup></em><strong>Λ</strong><em>A</em>∆<strong>p </strong>= <em>A<sup>T</sup></em><strong>Λ</strong><em>b</em></td>

   <td width="26">(19)</td>

  </tr>

  <tr>

   <td width="372">⇒ ∆<strong>p </strong>= (<em>A<sup>T</sup></em><strong>Λ</strong><em>A</em>)<sup>−1</sup><strong>Λ</strong><em>b</em></td>

   <td width="26">(20)</td>

  </tr>

 </tbody>

</table>

Here <strong>Λ </strong>is a diagonal matrix of weights corresponding to the residual term computed as per the choice of the robust M-estimator used, A is the jacobian matrix for each pixel of the template considered in the cost function, and b is the corresponding vector of residuals. Implement these two methods and test your new tracker on the car video sequence again (data/car/).

Implement these modifications for the forward compositional Lucas Kanade tracker that you implemented for Q2.1. Use the same function names with Robust appended. Include both versions of the code in your submission.

<h2>Q3.2x: LK Tracking on an Image Pyramid                                                               (20 points)</h2>

If the target being tracked moves a lot between frames, the LK tracker can break down. One way to mitigate this problem is to run the LK tracker on a set of image pyramids instead of a single image. The Pyramid tracker starts by performing tracking on a higher level (smaller image) to get a course alignment estimate, propagating this down into the lower level and repeating until a fine aligning warp has been found at the lowest level of the pyramid (the original sized image). In addition to being more robust, the pyramid version of the tracker is much faster because it needs to run fewer gradient descent iterations on the full scale image due to its coarse to fine approach.

Use the same function names as before but append ‘Pyramid’ to the function names while submitting your code. Include both versions of the code in your submission. Implement these modifications again for the forward compositional Lucas Kanade tracker on a flow warp.

<h1>4       Submission Summary</h1>

<ul>

 <li><strong>1 </strong>Derive the expression for the Jacobian Matrix</li>

 <li><strong>2 </strong>What is the computational complexity of inverse compositional method?</li>

 <li><strong>1 </strong>Write the forward compositional tracker (LK Tracker)</li>

 <li><strong>2 </strong>Initialize the inverse compositional tracker (MB Tracker)</li>

 <li><strong>3 </strong>Write the inverse compositional tracker (MB Tracker)</li>

 <li><strong>4 </strong>Run the inverse compositional tracker on the car dataset. What templates does work well with? When does the tracker break down? Why does this happen?</li>

 <li><strong>5 </strong>Run the inverse compositional tracker on the run markings dataset.</li>

 <li><strong>1x </strong>Add illumination robustness</li>

 <li><strong>2x </strong>LK Tracking on an image pyramid.</li>

</ul>

<h1>References</h1>

<ol>

 <li>Simon Baker, et al. Lucas-Kanade 20 Years On: A Unifying Framework: Part 1, CMURITR-02-16, Robotics Institute, Carnegie Mellon University, 2002</li>

 <li>Simon Baker, et al. Lucas-Kanade 20 Years On: A Unifying Framework: Part 2, CMURITR-03-35, Robotics Institute, Carnegie Mellon University, 2003</li>

 <li>Bouguet, Jean-Yves. Pyramidal Implementation of the Lucas Kanade Feature Tracker: Description of the algorithm, Intel Corporation, 2001</li>

</ol>