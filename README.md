test
# PipeNine-Complete Automation

## Overview

This project is a complete automation solution that includes a Spring Boot application and an older version of a Python Flask application. It leverages various tools and technologies such as Docker, Jenkins, Ansible, and Maven for building, deploying, and managing the application.

## Project Structure

The project structure is as follows:

- `.gitignore`
- `ansible/`
- `ansible.cfg`
- `inventory_aws_ec2.yaml`
- `my-playbook.yaml`
- `Dockerfile`
- `Jenkinsfile`
- `pom.xml`
- `prepare-ansible-server.sh`
- `python-flask-old-version/`
- `pycache/`
- `.dockerignore`
- `.DS_Store`
- `Dockerfile`
- `README.md`
- `requirements.txt`
- `server.py`
- `static/`
- `index.html`
- `README.md`
- `script.groovy`
- `src/`
    - `main/`
        - `java/`
            - `com/`
                - `sharansrj567/`
                    - `controller/`
                    - `model/`
                    - `security/`
                    - `service/`
        - `resources/`
            - `application.properties`
- `static/`
- `templates/`
- `test/`
    - `java/`
        - `com/`

## Spring Boot Application

### Building the Application

To build the Spring Boot application, use Maven:

```sh
mvn clean package -DskipTests
```

### Running the Application

To run the application, use Docker:

```sh
docker build -t your-image-name .
docker run -p 8080:8080 your-image-name
```

### Testing the Application

To run tests, use Maven:

```sh
mvn test
```

### Accessing the HTML Document

Once your Docker container is running, navigate to http://localhost:8000/index.html in your browser. This URL serves the index.html file, which makes API calls to the FastAPI backend running on the same port (8000) in the Docker container.

## Jenkins Pipeline

The Jenkins pipeline is defined in the `script.groovy` file. It includes stages for building the application, building the Docker image, and deploying the application.

## Jenkinsfile

The `Jenkinsfile` defines the pipeline stages and steps for building and deploying the application.

## Script Groovy

The `script.groovy` file contains the following functions:

- `buildJar()`: Builds the application using Maven.
- `buildImage()`: Builds and pushes the Docker image.
- `deployApp()`: Deploys the application.

## Ansible

The Ansible configuration is located in the `ansible` directory. It includes the `ansible.cfg`, `inventory_aws_ec2.yaml`, and `my-playbook.yaml` files for managing the deployment and configuration of the application.

## Configuration

### Application Properties

The Spring Boot application properties are defined in the `src/main/resources/application.properties` file.

### Dependencies

#### Spring Boot

The project uses the following Spring Boot dependencies:

- `spring-boot-starter-web`
- `thymeleaf-extras-springsecurity6`
- `jjwt`
- `spring-boot-devtools`
- `h2`
- `spring-boot-starter-test`
- `spring-security-test`
- `jaxb-api`
- `jaxb-runtime`

#### Python Flask

The Flask application uses the following dependencies:

- `fastapi`
- `pydantic`
- `uvicorn`
- `python-jose[cryptography]`
- `PyJWT`

## License

This project is licensed under the MIT License.

Feel free to customize this [`README.md`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2Ffiles%2FPipeNine-Complete%20Automation%2FREADME.md%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%223f6bf3c8-257c-4f84-b19e-4bfcb07f2899%22%5D "c:\files\PipeNine-Complete Automation\README.md") file further to suit your project's needs.