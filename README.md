# Dataset Toolbox design

## Overview
There are primarily two parts to the Dataset Toolbox: the set of python **Tools** that may be run from the CLI and the **NiceGUI** app that users can use to interact with to run the tools from a GUI.

## Tools
Currently there are three tools included in the toolbox.
1. An **OME converter** that converts `.ome.tif`/`.dmask.pgm` files to png images. The output images are organised acdording to their channel numbers.
2. A **RLE converter** that converts the RLE annotations returned by Label Studio into png images and masks.
3. A **Utilities** file that contains general purpose dataset tools. Applying overlays, viewing images etc.

## NiceGUI
There are three pages in the nicegui application.
1. A **Viewer** page that allows users to view their `.ome.tif` files with their default `dmask.pgm` masks. This page utilises the `def ome_reader` function from the **Utilities** tool.
2. A **Converter** page that allows user to convert their `.ome.tif` datasets into png images. This page is essentially a wrapper around the **OME converter** tool.
3. An **Editor** page that enables users to make fine-grained updates to their masks to ensure pixel perfect segmentation data. This page uses many utility functions provided in the **Utilities** tool. Due to its complexity it is only included as a NiceGUI page since CLI is not feasible for this use case.
4. TODO: A **RLE Converter** page that allows users to convert RLE annotations to png images and masks.
