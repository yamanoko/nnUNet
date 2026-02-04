#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import SimpleITK as sitk


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha',
        '.gipl'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if npy_image.ndim == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Read segmentation file. Supports both single-task (3D) and multi-task (4D) segmentations.
        
        For multi-task segmentation, the NIfTI file has shape (X, Y, Z, num_tasks) and SimpleITK
        reads it as (num_tasks, Z, Y, X). This method converts it to (num_tasks, X, Y, Z) format
        expected by nnU-Net.
        
        :param seg_fname: Path to segmentation file
        :return: Tuple of (segmentation array, properties dict)
        """
        itk_image = sitk.ReadImage(seg_fname)
        npy_seg = sitk.GetArrayFromImage(itk_image)
        spacing = itk_image.GetSpacing()
        
        if npy_seg.ndim == 2:
            # 2D segmentation
            npy_seg = npy_seg[None, None]
            max_spacing = max(spacing)
            spacing_for_nnunet = (max_spacing * 999, *list(spacing)[::-1])
        elif npy_seg.ndim == 3:
            # 3D single-task segmentation (standard case)
            npy_seg = npy_seg[None]
            spacing_for_nnunet = list(spacing)[::-1]
        elif npy_seg.ndim == 4:
            # 4D multi-task segmentation: SimpleITK returns (num_tasks, Z, Y, X)
            # This is already consistent with nnU-Net's image format (C, Z, Y, X)
            # No additional axis transformation needed
            spacing_for_nnunet = list(spacing)[::-1][1:]  # Exclude the 4th dimension spacing
        else:
            raise RuntimeError(f"Unexpected number of dimensions: {npy_seg.ndim} in segmentation file {seg_fname}")
        
        spacing_for_nnunet = list(np.abs(spacing_for_nnunet))
        
        properties = {
            'sitk_stuff': {
                'spacing': spacing,
                'origin': itk_image.GetOrigin(),
                'direction': itk_image.GetDirection()
            },
            'spacing': spacing_for_nnunet
        }
        return npy_seg.astype(np.float32), properties

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)


class SimpleITKIOWithReorient(SimpleITKIO):
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]], orientation = "RAS") -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(itk_image.GetDirection())
            itk_image = sitk.DICOMOrient(itk_image, orientation)
            # print(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(itk_image.GetDirection()))
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if npy_image.ndim == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0],
                'original_orientation': original_orientation
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict

    def read_seg(self, seg_fname: str, orientation: str = "RAS") -> Tuple[np.ndarray, dict]:
        """
        Read segmentation file with reorientation. Supports both single-task (3D) and multi-task (4D) segmentations.
        
        For multi-task segmentation, the NIfTI file has shape (X, Y, Z, num_tasks) and SimpleITK
        reads it as (num_tasks, Z, Y, X). This method converts it to (num_tasks, X, Y, Z) format
        expected by nnU-Net after applying reorientation.
        
        :param seg_fname: Path to segmentation file
        :param orientation: Target orientation (default: "RAS")
        :return: Tuple of (segmentation array, properties dict)
        """
        itk_image = sitk.ReadImage(seg_fname)
        original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(itk_image.GetDirection())
        itk_image = sitk.DICOMOrient(itk_image, orientation)
        
        npy_seg = sitk.GetArrayFromImage(itk_image)
        spacing = itk_image.GetSpacing()
        
        if npy_seg.ndim == 2:
            # 2D segmentation
            npy_seg = npy_seg[None, None]
            max_spacing = max(spacing)
            spacing_for_nnunet = (max_spacing * 999, *list(spacing)[::-1])
        elif npy_seg.ndim == 3:
            # 3D single-task segmentation
            npy_seg = npy_seg[None]
            spacing_for_nnunet = list(spacing)[::-1]
        elif npy_seg.ndim == 4:
            # 4D multi-task segmentation: SimpleITK returns (num_tasks, Z, Y, X)
            # This is already consistent with nnU-Net's image format (C, Z, Y, X)
            # No additional axis transformation needed
            spacing_for_nnunet = list(spacing)[::-1][1:]
        else:
            raise RuntimeError(f"Unexpected number of dimensions: {npy_seg.ndim} in segmentation file {seg_fname}")
        
        spacing_for_nnunet = list(np.abs(spacing_for_nnunet))
        
        properties = {
            'sitk_stuff': {
                'spacing': spacing,
                'origin': itk_image.GetOrigin(),
                'direction': itk_image.GetDirection(),
                'original_orientation': original_orientation
            },
            'spacing': spacing_for_nnunet
        }
        return npy_seg.astype(np.float32), properties

    def write_seg(self, seg, output_fname, properties):
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])
        itk_image = sitk.DICOMOrient(itk_image, properties['sitk_stuff']['original_orientation'])

        sitk.WriteImage(itk_image, output_fname, True)
