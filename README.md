# AI_Manufacturing_FA21
This is the github repository for the AI Manufacturing Project work done in the fall semester of 2021 under the direction of Dr. Vijay Gupta, Dr. Huachao Mao, and Bernardo Aquino Cruz.

[Frabutt AI Manufacturing Report FA21.pdf](https://github.com/jfrabut2/AI_Manufacturing_FA21/files/9620658/Frabutt.AI.Manufacturing.Report.FA21.pdf)

Jacob Frabutt
Dr. Vijay Gupta, Dr. Huachao Mao, and Bernardo Aquino Cruz
EE 28499: Undergraduate Research
3 December 2021

Machine Learning for Prediction of Geometry Deviation in Additive Manufacturing

Introduction
	The decreasing cost of printers and material, as well as the increasing array of printable materials have both been welcome advancements in the field of additive manufacturing. However, deformation of printed objects inhibits additive manufacturing from becoming a common, large-scale manufacturing solution. Prints need to be accurate to their design. Shrinkage or unexpected deformation is unacceptable in the production of components to be used in a medical or aviation setting, for example. Machine learning offers the potential to solve this problem. If a model can accurately predict how the deformation will occur, the printing instructions can be modified so that what gets printed by the 3D printer accurately renders the desired geometry. Under the direction of Dr. Vijay Gupta, Dr. Huachao Mao, and Bernardo Aquino Cruz, Alisa Nguyen and I worked to train a neural network that could successfully predict the deformation of given geometries.


Technical Description
	Our goal was to create a framework based on machine learning that could predict geometric deviations. Since producing large quantities of experimental data in additive manufacturing is costly and time consuming, we relied on simulated data provided by Dr. Mao. Ultimately, we sought to develop a model that integrated both simulated and experimental data that could output printing instructions such that no deviation would occur. To achieve this, we set the following three tasks:
 
Task 1: Understand and build upon the research team’s previously completed work. As the only person working on the team who did not do so last year, I had an increased responsibility at the beginning of the semester to review the work completed in the spring, familiarize myself with their code, and gain an understanding of the problem and neural networks in general. This background preparation was necessary for my later contributions to the project as the semester progressed.

Task 2: Refine and expand upon the neural network generated in previous semesters. We utilized MATLAB to create our neural networks, as well as using many of its toolboxes, such as the Deep Learning Toolbox, to draw upon pre-existing neural network structures and layers. Developing a deep neural network to accurately predict geometry deviations took up a large portion of our time.

Task 3: From the neural network that we constructed, attempt to solve the inverse problem, and use transfer learning. Concisely, the inverse problem consists of reversing the inputs and outputs of our model and training the network to predict in reverse. Transfer learning trains a network largely from simulated data, then completes the training with a smaller sample of real-world data. However, due to time constraints, we were unable to begin work on transfer learning.


Previous Work
	The team from the spring of 2021 considered an n×n×n field represented as a vector of size 1×n^3. An entry of 1 represented a voxel filled with material in it and an entry of 0 represented an empty voxel. The team then created a shallow neural network that would predict the deformation in the x, y, and z dimensions as a 1×3n^3 vector.

 
Figure 1. Previous dataset configuration of 1 sample. 

	With this approach, the team was able to create a shallow neural network that could accurately predict the deviation of each voxel in each direction. The team achieved excellent prediction results with this approach. The team then moved to creating a deep neural network. The team had to simplify the model to predict only the deviation in the x direction of the middle voxel, but they were able to obtain some promising results.


Results
	With encouraging results from the spring semester’s work, at the start of the fall semester the team wanted to shift to solving the inverse problem. That is, can we provide a network with the geometry we want printed, and receive as output the geometry that should be given as an instruction to the printer? Then, when the printer prints the generated instructions and the inevitable deformation occurs, we will be left with the desired geometry.

	The first step towards this goal was revisiting the data that the model would train from. Ideally, the input and output would both be n×n×n arrays, one representing the instruction geometry and the other representing the printed geometry. To reduce the complexity of our model, however, we simplified the problem to two dimensions. Dr. Mao created a data simulator in MATLAB. The simulator outputs random geometries in an “original” directory that represent the instruction geometries, and then corresponding shrunk geometries in a “distorted” directory that represent the printed geometries. Each sample is thus an original and corresponding distorted n×n array (stored as an image) where a pixel value of 0 represents an empty pixel and a pixel value of 255 represents a pixel with material.

 
Figure 2. New 2D dataset

	After obtaining a functional dataset generator, the next task was to solve the forward problem with this data. That is, can we train a neural network to predict the distortion of any given original image? We briefly considered the approach of using this new data with the previous semester’s shallow neural network. However, because the previous semester’s data was structured differently, and attempts to force the new data into the previous structure did not yield helpful results, this approach was quickly discarded.

	Instead, we began to work on implementing a U-Net structure for our neural network. A U-Net is a semantic segmentation network that takes in images and ultimately uses a pixel classification layer to predict the label of each pixel in an input image. One common application that we found of the U-Net is biomedical image segmentation (i.e., identifying a cancerous mass from an imaging scan). We used MATLAB’s function unetLayers() to generate a U-Net structure given the size of our images and the number of classes, which is 2 for our problem (background and shape). The function also allows you to adjust the number of layers the network has by specifying an encoder depth. As encoder depth increases, the U-Net gets “deeper,” and the network gains more layers. The values 4 (default) and 1 are displayed here. 

            
Figure 3. U-Net with default encoder depth		Figure 4. U-Net with decreased encoder 								depth for readability

	We created the structure of our U-Net and began training our network. We started with very small images (16x16 pixels) and achieved excellent validation results. However, the pictures at this size rarely had any shapes, so these results were trivial. Once we increased our image size to the point where there were substantial shapes in the images, our validation accuracy failed to produce the same promising results. As expected, the training accuracy always increased towards and neared 100%, and the training loss always decreased towards and neared 0. However, the validation accuracy always plateaued around 75% at its best, and the validation loss always began to shift away from 0. 

 

Figure 5. Training progress on a U-Net with 50 samples of 80x80 images


	We experimented with different variables to see if we could improve the validation results. We trained the network with larger images and used more samples, but neither fixed the issue. At the time, we were not sure of the exact issues with our network but thought it might be overfit. We adjusted the encoder depth, both increasing and decreasing it from its default value of 4. There were some small improvements, but no substantial fixes to the trends in validation accuracy and loss. After spending a few weeks on the U-Net, we were unable to find a solution to the sub-optimal validation performance, and we decided to investigate other approaches to our task.

	After doing some research into different network structures that might be compatible to our task, we began implementing a semantic segmentation network that used a Tversky Pixel Classification Layer. This layer used a value known as the Tversky index to calculate the loss over the number of classes in the two corresponding images. For mathematical definitions of these values and for a detailed description of MATLAB’s implementation of this layer, see the referenced site.  In our initial implementation of this approach, we found no improvement in the validation accuracy or loss. However, we eventually found an error in the division of our samples into training and validation sets, which had been negatively impacting our validation performance throughout the semester. Upon fixing this error, the validation accuracy correctly approached 100% and validation loss correctly approached 0.

	It was with this approach that we were first able to find a meaningful way to view what our model was predicting. Upon viewing our model’s output however, we realized that it was struggling to predict the correct shrinkage, despite the strong validation results. Figure 6 shows a collection of three images displaying the results. The first is the original image. The second is the output of the model, with the model’s predicted shrinkage shown as light blue. The final image displays the actual shrinkage in grey. It is clear here that the light blue does not match the grey as it should.

 
Figure 6. Results from semantic segmentation network with Tversky Pixel Classification Layer

	To improve the predictions of the model, we identified three possible changes. First, we used MATLAB’s experiment manager to find the optimal values of the hyperparameters that MATLAB uses when training the model. This produced only trivial improvements in visible accuracy. Next, we experimented with the filter sizes and the number of filters in our convolutional layers. This too did not produce substantial improvements. Finally, we investigated the parameters of the Tversky loss function, but still did not generate promising results.

	From the model’s output it was clear that the network was understanding the general shape of the input, but it was not getting a “global” sense of how the shrinkage should occur. This need for considering more global information led us back to the U-Net structure. The error that had been causing the poor validation performance was fixed (i.e., correct division of training and validation sets), so our revisited implementation of the U-Net quickly yielded very promising results. We continued to train the network using more samples, as well as testing with larger images. The model produced good performance with larger images, but we worked mostly with 80x80 pixel images because these had a good balance of image complexity with shorter training time. Figure 7 displays the original picture in the top left corner, the model’s shrinkage in light blue in the top right corner, the model’s error in either light or dark blue in the bottom left corner, and the ground truth distortion in the bottom right corner. A confusion matrix for the testing data is also shown.

 
Figure 7. U-Net, 80x80 pixels, 50,000 samples


 
Figure 8. Confusion matrix for the testing data of the same net trained in Figure 7


	Finally, after training the U-Net many times with various sample sizes and image sizes and obtaining excellent results on those tests, we moved to looking at the inverse problem. We considered two approaches to training the inverse neural network. One method was to simply switch the inputs and outputs in our code and train the new network. The other method was to use a fully trained forward network to generate as many distorted samples as we wanted, and then feed those in as inputs with the ground truth originals as outputs. We were able to get both methods to work, and both achieved good results, but the simpler method of swapping the inputs and outputs outperformed the latter approach in validation accuracy, the visual output, and in the number of false positives and negatives in the confusion matrix. Figure 9 presents the results from the first method. Since the inputs and outputs are swapped, the distorted picture is now in the top left corner and the original is in the bottom right. The top right still shows the model’s prediction, and the bottom left shows its error.

 
Figure 9. Inverse U-Net, 80x80 pixels, 10,000 samples


 
Figure 10. Confusion matrix for the testing data of the same net trained in Figure 9


Discussion and Conclusion
	Deformation continues to be an obstacle to the large-scale success of additive manufacturing. Factors such as heat, material, and machine specific differences all can lead to variations in the geometry of printed objects. Machine learning offers a potential solution to this problem.
	Our research this semester was successful in that it upheld and strengthened the previous semester’s finding that machine learning can be used to accurately predict the geometric deviations in additive manufacturing. Furthermore, the inverse of this problem, where the model outputs the shape to be printed to obtain a desired geometry, can be accurately predicted by machine learning as well.
	One way that the model could be improved is to incorporate transfer learning as originally planned. A small amount of real-world experimental data could be used to supplement the large amounts of simulated data that can be generated to create a model that will accurately predict the deformations of physical geometries, not just simulated ones. Another important step forward would be to expand the model to three dimensions. We worked with two dimensions to simplify the task as we tried to implement a functioning neural network. Now that the model has been successfully trained with two dimensions, it makes sense to move closer to the actual physical problem and consider three dimensions.
	For review and reference, the code to reproduce the results of both the forward and inverse U-Nets can be found on our GitHub repository.

![image](https://user-images.githubusercontent.com/89485604/191617142-44504999-111f-4e71-b3cf-1a908dfecea1.png)

