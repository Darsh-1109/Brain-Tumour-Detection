import React, { useState, useEffect } from "react";
import "./Home.css";
// import { useNavigate } from 'react-router-dom';

function DoctorPage() {
  const [files, setFiles] = useState([]);
  const [fileDescription, setFileDescription] = useState([]);
  const [senderName, setSenderName] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  //   const navigate = useNavigate();

  const navigateHome = () => {
    navigate("/");
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = () => {
    fetch("http://127.0.0.1:5000/tumour_get_uploaded_files")
      .then((response) => response.json())
      .then((data) => {
        // Sort the files array by date_created in ascending order
        const sortedFiles = data.sort(
          (a, b) => new Date(a.date_created) - new Date(b.date_created)
        );
        // Assign a displayId to each file for display purposes
        const filesWithDisplayId = sortedFiles.map((file, index) => ({
          ...file,
          displayId: index + 1,
        }));
        setFiles(filesWithDisplayId);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  };

  const handleDelete = (displayId, id) => {
    fetch(`http://127.0.0.1:5000/tumour_delete_file/${id}`, {
      method: "DELETE",
    })
      .then((response) => {
        if (response.ok) {
          // File deleted successfully; refresh the file list
          fetchFiles();
        } else {
          console.error("Failed to delete file.");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };

  const handleUpload = () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("description", fileDescription);
      formData.append("sender_name", senderName);

      // Send the FormData to your Flask backend using the fetch API or Axios.
      fetch("http://127.0.0.1:5000/tumour_upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          if (response.ok) {
            console.log("File uploaded successfully.");
            // After uploading, fetch the updated file list
            fetchFiles();
          } else {
            console.error("Failed to upload file.");
          }
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    }
  };

  return (
    <div>
      <h2 className="heading">Brain Tumour Detection</h2>
      <div className="inputFieldsContainer">
        <input
          className="inputField"
          id="patientTextField"
          type="text"
          placeholder="Enter Patient Name"
          value={senderName}
          onChange={(e) => setSenderName(e.target.value)}
        />
        <input
          className="inputField"
          id="descriptionTextField"
          type="text"
          placeholder="Enter Description"
          value={fileDescription}
          onChange={(e) => setFileDescription(e.target.value)}
        />
      </div>
      
      <div className="uploadButtonsContainer">
      {/* <label style={{ fontWeight: "bold" }} for="fileInput">
          Upload MRI Scan:{" "}
        </label> */}
      <input
        id="fileInput"
        type="file"
        accept=".jpg, .jpeg, .png, .pdf, .doc, .docx"
        onChange={handleFileChange}
      />
      <button onClick={handleUpload}>Upload File</button>
      </div>

      <p className="tableName">Patient MRI Scans</p>
      <div className="tableContainer">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Filename</th>
              <th>Patient Name</th>
              <th>Description</th>
              <th>Tumour Type</th>
              <th>Affected Body Functionality</th>
              <th>Segmented Image</th>
              <th>Date Uploaded</th>
            </tr>
          </thead>
          <tbody>
            {files.map((file) => (
              <tr key={file.displayId}>
                <td>{file.displayId}</td>
                <td>
                  <a
                    href={`http://127.0.0.1:5000/tumour_uploads/${file.filename}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {file.filename}
                  </a>
                </td>
                <td>{file.sender_name}</td>
                <td>{file.description}</td>
                <td>{file.tumour_type}</td>
                <td>{file.affected_body_functionality}</td>
                <td>
                  <a
                    href={`http://127.0.0.1:5000/segmented_uploads/${file.segmented_image}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {/* {file.segmented_image} */}
                    <img
                      src={`http://127.0.0.1:5000/segmented_uploads/${file.segmented_image}`}
                      style={{
                        maxWidth: "100%",
                        maxHeight: "100%",
                        height: "auto",
                      }}
                      alt="segmented_image"
                    />
                  </a>
                </td>
                <td>{file.date_created}</td>
                <td>
                  <button onClick={() => handleDelete(file.displayId, file.id)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default DoctorPage;
