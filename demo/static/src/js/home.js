import axios from 'axios';
import React from 'react';

import {
    Button,
    Card, CardActions,
    CardContent,
    Grid,
    List, ListItem, ListItemIcon,
    Typography
} from '@material-ui/core';

import * as ReactDOM from "react-dom";
import {AttachFile} from "@material-ui/icons";

class LoginUI extends React.Component {

    state = {
        selectedFile: null,
        prediction: null
    };

    // Updating the selectedFile when the user selects a file
    onFileChange = event => {
        this.setState({ selectedFile: event.target.files[0] });
        this.setState( {prediction: null})

        let reader = new FileReader();
        reader.onload = (e) => {
            this.setState({image: e.target.result});
        };
        reader.readAsDataURL(event.target.files[0]);
    };

    // Displaying file information when it is uploaded
    onFileUpload = () => {
        const formData = new FormData();

        // Update the formData object
        formData.append(
            "user_image",
            this.state.selectedFile,
            this.state.selectedFile.name
        );

        const headers = {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        };
        console.log(this.state.selectedFile);
        axios.post("http://localhost:8855/upload_file", formData, headers)
            .then(res => {
                console.log({res});
                console.log(res.data["emotion"])
                this.setState({prediction: res.data["emotion"]})
            })
            .catch(err => {
                console.error({err});
            });

    };

    fileData = () => {
        if (this.state.selectedFile) {
            return (
                <div>
                    <Typography variant="h4" align="left">
                        File Details
                    </Typography>
                    <Typography variant="body1">File Name: {this.state.selectedFile.name}</Typography>
                    <Typography variant="body1">File Type: {this.state.selectedFile.type}</Typography>
                    <Typography variant="body1">Last Modified:{" "}
                        {this.state.selectedFile.lastModifiedDate.toDateString()}
                    </Typography>
                </div>
            );
        }
    };

    constructor(props) {
        super(props);
        this.state = {
            isFetching: true,
            users: []
        };
    }

    render() {
        const style = {
            width: '100%',
            boxSizing: 'border-box',
            padding: '9px',
            resize: 'none',
            fontSize: '18px'
        };
        return (
            <React.Fragment>
                <Grid
                    container
                    direction="column"
                    justify="space-between"
                    alignItems="center"
                >
                    <Grid
                        container
                        item
                        justify="center"
                        alignItems="center"
                        xs={8}
                    >
                        <Card>
                            <CardContent>
                                <Typography variant="h3" align="center">
                                    Predict from Image
                                </Typography>
                                <List
                                    component="nav">
                                    <ListItem>
                                        <ListItemIcon>
                                            <AttachFile/>
                                        </ListItemIcon>
                                        <Button
                                            component="label"
                                            size={"small"}
                                            color="primary"
                                        >
                                            Choose Image
                                            <input type="file" onChange={this.onFileChange} hidden/>
                                        </Button>
                                    </ListItem>
                                    <ListItem>
                                        {this.fileData()}
                                    </ListItem>
                                </List>
                                <img id="target" src={this.state.image} width="100%"/>
                                <Typography variant="h4">
                                    {(this.state.prediction != null) ? "Prediction: " : ""} {this.state.prediction}
                                </Typography>
                            </CardContent>
                            <CardActions>
                                <Button
                                    variant={"contained"}
                                    size={"small"}
                                    color="primary"
                                    onClick={this.onFileUpload}>
                                    Run
                                </Button>
                            </CardActions>
                        </Card>
                    </Grid>
                </Grid>
            </React.Fragment>
        );
    }
}

const domContainer = document.querySelector('#login_container');
ReactDOM.render(<LoginUI/>, domContainer);