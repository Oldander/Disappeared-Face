

# Disappeared Face: A Physical Adversarial Attack Method on Black-Box Face Detection Models

## Abstract 

Face detection is a classical problem in the field of computer vision. It has significant application value in face-scan payment, identity authentication, image focus, and other areas. The emergence of adversarial algorithms on face detection poses a substantial threat to the security of face detection. The current adversarial attack on face detection mainly focuses on white-box attacks. Due to the limitations of the need to fully understand the face detection model structure and corresponding parameters, the white-box adversarial attack’s transferability, which can measure the attack’s effectiveness across many other models, is not high. What's more, the perturbation loss caused by complex physical environments like light and noises sometimes prevents existing white-box attack methods from taking effect in the real world. Aiming at solving the above problems, we propose a Black-Box Physical Attack Method on face detection. Through ensemble learning, we can extract the public attention heat map of the face detection models. The attack against the public attention heat map has high transferability across models. Our method realizes the successful escape of both the white-box and black-box face detection models in both the PC terminal and the mobile terminal, including the camera module, mobile payment module, selfie beauty module, and official face detection models.



The attack is implemented in **adversarial_patch_gen_4models.py** source file, in order to train your own patches to attack commercial Black-Box face detection models in the real world, follow the guideline:
1. Print the 'Checkerboards.docx' with a laser printer and then post them on your face.
2. Take pictures under different conditions (light, distance).
3. Set images (I use 8 face images of myself):
Use pixels to calibrate the mask area.
4. Specify patches parameters;
5. Specify losses(You can add any loss function defined by different face detection models to the total loss function,we select all four models' loss functions here).
6. run 'adversarial_patches_gen_4models.py' and obtain output_imgs.
7. Print the 'left_cheek.png' and 'right_cheek.png' with a laser printer  and then post them on your face.
8. Sit in front of the camera and try to escape the black-box face-detection model in the real world.


