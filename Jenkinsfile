#!/usr/bin/env groovy

pipeline {
    agent any
    environment {
        ANSIBLE_SERVER = "123.456.789.012"      
    }
    stages {
        stage("copy files to ansible server") {
            steps {
                script {
                    echo "copying all necessary files to ansible server"
                    sshagent(['ansible-server-key']) {
                        sh "scp -o StrictHostKeyChecking=no ansible/* root@${ANSIBLE_SERVER}:/root"

                        withCredentials([sshUserPrivateKey(credentialsId: "ec2-server-key", keyFileVariable: 'keyfile', usernameVariable: 'user')]) {
                            sh 'scp $keyfile root@$ANSIBLE_SERVER:/root/ssh-key.pem'
                        }
                    }
                }
            }
        }

        stage("execute ansible playbook") {
            steps {
                script {
                    echo "calling ansible playbook to configure ec2 instances"
                    def remote = [:]
                    remote.name = "ansible-server"
                    remote.host= env.ANSIBLE_SERVER
                    remote.allowAnyHosts = true

                    withCredentials([sshUserPrivateKey(credentialsId: "ansible-server-key", keyFileVariable: 'keyfile', usernameVariable: 'user')]) {
                        remote.user = user
                        remote.identityFile = keyfile
                        // Optional
                        // sshScript remote: remote, script: "prepare-ansible-server.sh"
                        sshCommand remote: remote, command: "ansible-playbook my-playbook.yaml"
                    }
                }
            }
        }
    }
}