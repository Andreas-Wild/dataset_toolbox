This document is a guide for setting up Label Studio with a ML backend (Specifically we will use [SAM2](https://github.com/facebookresearch/sam2)).

## Prerequisites:
- Docker
- GPU accelerated environment (i.e the command nvidia-smi and nvcc --version should both work)

## Setup Script:
- A setup script is included. Skip to step 3 if used. In general, running random bash scripts is a bad idea. Only do this if you trust the author or have inspected the script yourself.

```bash
chmod +x setup.sh && sudo bash setup.sh
```
## Step 1: Clone the repository

```bash
git clone https://github.com/Andreas-Wild/ls_and_ml_backend.git
```
## Step 2: Create the necessary folders/files

```bash
sudo mkdir files data sam2
cd sam2 && git clone https://github.com/HumanSignal/label-studio-ml-backend.git && cd ..
sudo nano .env
```

Inside the `.env` file you will need to add a Label Studio Legacy token, available at step 4.

```bash
LS_API_KEY=<enter your token here>
```

Since Docker runs as a non-root user, we need to ensure that it has read permissions.
```bash
sudo chown -R 1001:1001 data files
```
## Step 3: Run docker compose

```bash
docker compose up --build
```
This will build the container for the first time. This may take a while (10-30 mins) as there are some heavy dependencies being used here:
- Pytorch
- CUDA
- Label Studio
- SAM2
- etc.
Don't worry this only happens once. Subsequent restarts will be much faster.
## Step 4: Get your legacy API Token

1. In your browser of choice navigate to `http://localhost:8080/` .
2. Click on the "Label Studio" logo and navigate to "Organization"
3. Click on "API Tokens Settings".
4. Select "Legacy Tokens" and "Save Changes".
5. Now click on your user in the top right of the screen. Enter "Account & Settings".
6. Navigate to "Legacy Token" and copy your "Access Token".
7. Paste this token into the `.env` file you created.
8. CTRL+C in your terminal to stop the container. (The environment has changed and now contains your legacy access token)
9. Restart the services with `docker compose up`

### Optional Tip
Use `docker compose up -d` to run the container in detached mode if you would like to have control over your terminal again.
To stop the containers in detached mode use `docker compose down`.

## Step 5: Create a labelling project

1. Click on "Create Project" and name your project.
2. Don't change anything under Data Import.
3. For your labelling setup, click on custom template and paste in the following XML file.

```xml title="label_interface_template.xml"
<View>
  <Style>
    .ls-main-content { background-color: #f0f2f5; }
    .ls-frames { border: 2px solid #ccc; border-radius: 8px; }
  </Style>
!Here you can prompt the model backend
    <Header value="SAM2"/>
  
  <RectangleLabels name="prompts_bbox" toName="image" smart="true">
    <Label value="Object_Type_A" background="#FF4D4F"/>
    <Label value="Object_Type_B" background="#52C41A"/>
    <Label value="Object_Type_C" background="#1890FF"/>
  </RectangleLabels>

  <KeyPointLabels name="prompts_points" toName="image" smart="true">
    <Label value="Object_Type_A" background="#FF4D4F"/>
    <Label value="Object_Type_B" background="#52C41A"/>
    <Label value="Object_Type_C" background="#1890FF"/>
    <Label value="Negative" background="#000000"/>
  </KeyPointLabels>

  <Image name="image" value="$image" zoom="true" scrollBar="true"/>
!SAM2 returns these types of labels
  <Header value="Manually use brush labels:"/>
  
  <BrushLabels name="masks" toName="image">
    <Label value="Object_Type_A" background="#FF4D4F"/>
    <Label value="Object_Type_B" background="#52C41A"/>
    <Label value="Object_Type_C" background="#1890FF"/>
  </BrushLabels>


</View>
```

4. Change the labels to match your specific project and click "Save".
5. Select the project and enter the "Settings" in the toolbar tab.
6. Under "Source Cloud Storage", click on "Add Source Storage".
7. Select "Local Files" and "Next".
8. Give the storage a name and add the absolute path to the storage.
    - Choosing the correct path:
    - Since we are using docker containers we need to choose the correct path. 
    - For example, a labelling task about dogs should be saved under `./files/dogs/dog1.jpg`. 
    - Inside the "Absolute local path" field enter the following `/label-studio/files/dogs/`.
    - This is required since we mounted our local folder into our container as:
    `./files = /label-studio/files`
    The images are always stored locally, the container simply reads them.

9. Choose "Files" as your "Import Method" and click on "Images" to add a regex pattern to filter files inside the directory. Load the preview and ensure you can see your images.
10. Click "Save and Sync".
## Step 6: Add the ML backend
1. Still inside your project settings select "Model".
2. Click on "Connect Model" and enter a name for the model.
3. Enter the "Backend URL" as `http://sam2-ml-backend:9090`
    - Choosing the correct URL
    - Since we are using one docker-compose.yml file to orchestrate the containers, the containers can see each other over the Docker network. So we can use the **docker service name** inside the URL.
    - If we didn't have this docker network we would use: `http://host.docker.internal:9090`.
    - See more information about this, [here](https://labelstud.io/guide/ml#localhost-and-Docker-containers).
    - If both Label Studio and the ML backend were running locally we could simply use: `http://localhost:9090`

4. Select "Interactive pre-annotations" and select "Validate and Save".
5. Check that the model says "Connected".
## Step 7: Start labelling!
Now the data is loaded and the model is connected, we are ready to start labelling.
