
<?php

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['csv_file'])) {
    $file = $_FILES['csv_file'];

    if ($file['error'] === UPLOAD_ERR_OK) {
        $uploaded_file = $file['name'];
        $destination = '/tmp/' . $uploaded_file;

        if(!move_uploaded_file($file['tmp_name'], $destination)) {
            echo "Error moving uploaded file.";
            exit;
        }
        $python_script = './script.py'; // Replace with the actual path
        $venv_activate = '/home/debian/textmining_ve/bin/activate';
        $command = "/bin/bash -c 'source $venv_activate && python3 $python_script $destination ./'"; 
        $output = shell_exec($command);

        echo $output;
        exit;
    } else {
        echo "Error uploading file. Error code: " . $file['error'];
    }
}

?>

