import torchio as tio
from pathlib import Path

class SubjectBuilder:

    @staticmethod
    def build_subjects(
        subjects_dir,
        modalities=("t1c", "t1n", "t2w   ", "t2f"),
        seg_token="seg"
    ):
        """
        subjects_dir should be a path to the top level folder containing subject sub folders the e.g.
            user/project/glioseg/data/{subjects_dir}/
                subject_001/
                    *-t1n.nii.gz
                    *-t1c.nii.gz
                    *-t2w.nii.gz
                    *-t2f.nii.gz
                    *-seg.nii.gz
                subject_002/
        
        modalities defines the image labels e.g. t1c for *-t1c.nii.gz, t1n for *-t1n.nii.gz

        seg_token defines the token associated with segmentation masks within the sub-folder
        """


        subjects = []
        
        modalities = tuple(m.strip().lower() for m in modalities)
        seg_token = seg_token.strip().lower()

        for subject_dir in subjects_dir.iterdir():

            #skips non dir objects in the top level
            if not subject_dir.is_dir():
                continue

            nii_files = list(subject_dir.glob("*.nii.gz"))
            image_dict = {"subject_id": subject_dir.name}
            #loops over 
            for mod in modalities:
                match = next(
                    (image for image in nii_files if image.name.lower().endswith(f"-{mod}.nii.gz")),
                    None,
                )
                if match is not None:
                    image_dict[mod] = tio.ScalarImage(match)

            seg_match = next(subject_dir.glob("*-seg.nii.gz"), None)
            if seg_match is not None:
                image_dict["seg"] = tio.LabelMap(seg_match)
            
            if len(image_dict) > 1:
                subjects.append(tio.Subject(**image_dict))

        return subjects

